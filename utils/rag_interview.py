"""
rag_interview.py
=================
Implements the "Interview Knowledge Base" chatbot. This module loads
all PDF guides stored in `data/interview_questions/` (e.g. "Java
Interview Guide.pdf", "Python Guide.pdf", "SQL Guide.pdf", "Machine
Learning Guide.pdf"), builds a FAISS index over them, and answers
interview-prep questions grounded ONLY in that uploaded knowledge
base - useful for HR/interviewers to quickly look up good questions
or reference answers on a given topic.

Reuses the shared chunking / embedding / FAISS / answer-generation
helpers defined in rag_resume.py to avoid duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss

from utils.rag_resume import chunk_text, build_faiss_index, search_index, answer_from_context
from utils.pdf_parser import extract_text_from_path
from utils.config import INTERVIEW_KB_DIR, get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this information in the interview knowledge base."


def list_knowledge_base_files() -> List[Path]:
    """
    List all PDF files currently stored in the interview knowledge
    base folder (data/interview_questions/).

    Returns:
        A list of Path objects pointing to each PDF file.
    """
    return sorted(INTERVIEW_KB_DIR.glob("*.pdf"))


@dataclass
class InterviewKnowledgeBaseRAGEngine:
    """
    A RAG engine that indexes all PDFs in the interview knowledge
    base folder and answers questions grounded only in that content.

    Typical usage (in Streamlit):
        engine = InterviewKnowledgeBaseRAGEngine()
        engine.build_index_from_folder()
        answer = engine.ask("What are common SQL join interview questions?")
    """
    chunks: List[str] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)
    _index: faiss.Index | None = field(default=None, repr=False)

    def build_index_from_folder(self) -> int:
        """
        Load every PDF in the interview knowledge base folder,
        extract its text, and build a combined FAISS index.

        Returns:
            The number of PDF files successfully indexed.
        """
        pdf_files = list_knowledge_base_files()

        if not pdf_files:
            logger.warning(
                "No PDFs found in %s. Upload guides like 'Python Guide.pdf' "
                "to enable the Interview Knowledge Base chatbot.",
                INTERVIEW_KB_DIR,
            )
            return 0

        self.chunks = []
        self.chunk_sources = []

        indexed_count = 0
        for pdf_path in pdf_files:
            try:
                text = extract_text_from_path(str(pdf_path))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to read %s: %s", pdf_path.name, exc)
                continue

            if not text:
                continue

            file_chunks = chunk_text(text, chunk_size=300, overlap=50)
            self.chunks.extend(file_chunks)
            self.chunk_sources.extend([pdf_path.name] * len(file_chunks))
            indexed_count += 1

        if not self.chunks:
            logger.warning("No extractable text found across knowledge base PDFs.")
            return 0

        self._index, _ = build_faiss_index(self.chunks)
        logger.info(
            "Indexed %d knowledge-base PDF(s) into %d chunks.",
            indexed_count, len(self.chunks),
        )
        return indexed_count

    def add_document(self, filename: str, text: str) -> None:
        """
        Add a single new document to an already-built index (used
        when HR uploads a new guide during the session without
        re-scanning the whole folder).

        Args:
            filename: Display name of the document (e.g. "SQL Guide.pdf").
            text: Extracted text of the document.
        """
        new_chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.chunks.extend(new_chunks)
        self.chunk_sources.extend([filename] * len(new_chunks))
        self._index, _ = build_faiss_index(self.chunks)
        logger.info("Added '%s' to the interview knowledge base index.", filename)

    def ask(self, question: str, top_k: int = 4) -> str:
        """
        Ask a natural-language question about the interview
        knowledge base.

        Args:
            question: The user's question (e.g. "Give me SQL join questions").
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
            domain_description="interview preparation knowledge base",
        )
