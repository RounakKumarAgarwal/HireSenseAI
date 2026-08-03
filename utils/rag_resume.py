"""
rag_resume.py
==============
Implements Retrieval-Augmented Generation (RAG) for the "Resume
Chatbot" feature - lets HR ask natural-language questions about a
specific candidate's resume (e.g. "Does this candidate know SQL?",
"How many years of experience?") and get answers grounded ONLY in
that resume's text.

This module also defines the SHARED RAG building blocks that
rag_policy.py and rag_interview.py reuse:
    - get_embedding_model()  : cached SentenceTransformers loader
    - chunk_text()             : splits long text into overlapping chunks
    - build_faiss_index()      : builds an in-memory FAISS vector index
    - search_index()           : retrieves the top-k most relevant chunks
    - answer_from_context()    : asks Groq to answer using ONLY the
                                  retrieved context, with a graceful
                                  "not found" fallback

Why RAG instead of just pasting the whole resume into the prompt?
    RAG (retrieve only the most relevant chunks, then generate an
    answer) keeps the LLM prompt small and focused, scales to very
    long documents (e.g. multi-page policy PDFs), and - most
    importantly - reduces hallucination by grounding every answer in
    real retrieved text rather than the model's memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils.summarizer import generate_completion
from utils.config import EMBEDDING_MODEL_NAME, get_logger

logger = get_logger(__name__)

_embedding_model: SentenceTransformer | None = None

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded resume."


# ------------------------------------------------------------------
# Shared RAG building blocks (reused by rag_policy.py & rag_interview.py)
# ------------------------------------------------------------------

def get_embedding_model() -> SentenceTransformer:
    """
    Lazily load (and cache) the SentenceTransformers embedding model
    so it's only downloaded/loaded into memory once per app session.

    Returns:
        A cached SentenceTransformer instance.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """
    Split a long text into overlapping word-based chunks, suitable
    for embedding and semantic search.

    Args:
        text: The full document text to split.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words shared between consecutive chunks
            (helps avoid losing context at chunk boundaries).

    Returns:
        A list of text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_faiss_index(chunks: List[str]) -> tuple[faiss.Index, np.ndarray]:
    """
    Embed a list of text chunks and build an in-memory FAISS index
    for fast similarity search over them.

    Args:
        chunks: List of text chunks to index.

    Returns:
        A tuple of (faiss_index, embeddings_array).
    """
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim (since normalized)
    index.add(embeddings.astype("float32"))

    logger.info("Built FAISS index with %d chunks (dim=%d).", len(chunks), dimension)
    return index, embeddings


def search_index(
    index: faiss.Index,
    chunks: List[str],
    query: str,
    top_k: int = 3,
) -> List[str]:
    """
    Retrieve the top-k most semantically relevant chunks for a query.

    Args:
        index: A pre-built FAISS index.
        chunks: The original text chunks (parallel to the index).
        query: The user's natural-language question.
        top_k: Number of top chunks to retrieve.

    Returns:
        A list of the most relevant text chunks, best match first.
    """
    if index.ntotal == 0 or not chunks:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    k = min(top_k, len(chunks))
    _distances, indices = index.search(query_embedding.astype("float32"), k)

    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def answer_from_context(
    query: str,
    context_chunks: List[str],
    not_found_message: str = NOT_FOUND_MESSAGE,
    domain_description: str = "document",
) -> str:
    """
    Ask the Groq LLM to answer a question using ONLY the provided
    context chunks. If the answer isn't in the context, the LLM is
    instructed to say so explicitly.

    Args:
        query: The user's question.
        context_chunks: Retrieved context chunks relevant to the query.
        not_found_message: Exact fallback message to use when the
            answer cannot be found in the context.
        domain_description: Short description of what the context
            represents (used in the prompt, e.g. "resume", "company
            policy document", "interview knowledge base").

    Returns:
        The LLM's grounded answer as plain text.
    """
    if not context_chunks:
        return not_found_message

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        f"You are a precise assistant that answers questions using "
        f"ONLY the provided {domain_description} context. Never use "
        f"outside knowledge or make assumptions. If the answer is not "
        f"clearly present in the context, respond with EXACTLY this "
        f'sentence and nothing else: "{not_found_message}"'
    )

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question using only the context above."
    )

    return generate_completion(system_prompt, user_prompt, temperature=0.2, max_tokens=400)


# ------------------------------------------------------------------
# Resume-specific chatbot
# ------------------------------------------------------------------

@dataclass
class ResumeRAGEngine:
    """
    A per-candidate RAG engine that indexes one resume's text and
    answers questions grounded only in that resume.

    Typical usage (in Streamlit):
        engine = ResumeRAGEngine(candidate_name="John Doe")
        engine.index_resume(resume_text)
        answer = engine.ask("Does this candidate know SQL?")
    """
    candidate_name: str = "Candidate"
    chunks: List[str] = field(default_factory=list)
    _index: faiss.Index | None = field(default=None, repr=False)

    def index_resume(self, resume_text: str) -> None:
        """Chunk and embed a resume's text, building its FAISS index."""
        self.chunks = chunk_text(resume_text, chunk_size=250, overlap=40)
        if not self.chunks:
            logger.warning("No text available to index for %s.", self.candidate_name)
            return
        self._index, _ = build_faiss_index(self.chunks)
        logger.info("Indexed resume for %s (%d chunks).", self.candidate_name, len(self.chunks))

    def ask(self, question: str, top_k: int = 3) -> str:
        """
        Ask a natural-language question about the indexed resume.

        Args:
            question: The HR user's question.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            A grounded answer, or the standard "not found" message.
        """
        if self._index is None or not self.chunks:
            return NOT_FOUND_MESSAGE

        relevant_chunks = search_index(self._index, self.chunks, question, top_k=top_k)
        return answer_from_context(
            question, relevant_chunks,
            not_found_message=NOT_FOUND_MESSAGE,
            domain_description=f"resume of {self.candidate_name}",
        )
