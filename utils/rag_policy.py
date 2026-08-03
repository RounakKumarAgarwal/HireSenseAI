"""
rag_policy.py
=============
Implements the "Company Policy Chatbot" feature. HR can upload one or
more company policy PDFs (e.g. leave policy, code of conduct, WFH
policy), and this module builds a FAISS vector index over them so HR
staff can ask natural-language questions and get answers grounded
ONLY in the uploaded policy documents.

Reuses the shared chunking / embedding / FAISS / answer-generation
helpers defined in rag_resume.py to avoid duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import faiss

from utils.rag_resume import chunk_text, build_faiss_index, search_index, answer_from_context
from utils.config import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded company policy documents."


@dataclass
class PolicyRAGEngine:
    """
    A RAG engine that indexes one or more company policy documents
    and answers HR questions grounded only in those documents.

    Typical usage (in Streamlit):
        engine = PolicyRAGEngine()
        engine.index_policies({"Leave Policy.pdf": policy_text_1,
                                "Code of Conduct.pdf": policy_text_2})
        answer = engine.ask("How many paid leaves do employees get?")
    """
    chunks: List[str] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)  # parallel list: which file each chunk came from
    _index: faiss.Index | None = field(default=None, repr=False)

    def index_policies(self, policy_documents: dict[str, str]) -> None:
        """
        Chunk, embed, and index one or more policy documents.

        Args:
            policy_documents: Dictionary mapping
                {filename: extracted_text} for each uploaded policy PDF.
        """
        self.chunks = []
        self.chunk_sources = []

        for filename, text in policy_documents.items():
            file_chunks = chunk_text(text, chunk_size=300, overlap=50)
            self.chunks.extend(file_chunks)
            self.chunk_sources.extend([filename] * len(file_chunks))

        if not self.chunks:
            logger.warning("No policy text available to index.")
            return

        self._index, _ = build_faiss_index(self.chunks)
        logger.info(
            "Indexed %d policy document(s) into %d chunks.",
            len(policy_documents), len(self.chunks),
        )

    def ask(self, question: str, top_k: int = 4) -> str:
        """
        Ask a natural-language question about the indexed policies.

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
            domain_description="company policy document",
        )
