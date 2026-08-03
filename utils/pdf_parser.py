"""
pdf_parser.py
=============
Handles all PDF text-extraction logic for HireSense AI.

Two libraries are used together for maximum reliability:
    - pdfplumber : Great at preserving layout / whitespace, used as
                    the PRIMARY extraction method.
    - PyPDF2      : Lightweight fallback used if pdfplumber fails
                    (e.g. on malformed or unusual PDFs).

Why two libraries?
    Real-world resumes come from many different tools (Word, Canva,
    LaTeX, LinkedIn exports, scanned PDFs, etc.) and no single parser
    handles 100% of them well. Falling back to a second parser makes
    text extraction far more robust for a college project demo.
"""

from __future__ import annotations

import io
from typing import Union

import pdfplumber
from PyPDF2 import PdfReader

from utils.config import get_logger

logger = get_logger(__name__)


def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    """Try extracting text using pdfplumber (primary method)."""
    text_chunks: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def _extract_with_pypdf2(file_bytes: bytes) -> str:
    """Try extracting text using PyPDF2 (fallback method)."""
    text_chunks: list[str] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def extract_text_from_pdf(uploaded_file: Union[bytes, "io.BufferedReader", object]) -> str:
    """
    Extract raw text from a PDF file.

    Args:
        uploaded_file: Either raw bytes, a file-like object (such as
            a Streamlit UploadedFile), or a path-like object with a
            .read() method.

    Returns:
        The extracted text as a single string. Returns an empty
        string if extraction completely fails (e.g. scanned image
        PDF with no embedded text layer).
    """
    # Normalise input to raw bytes
    if isinstance(uploaded_file, (bytes, bytearray)):
        file_bytes = bytes(uploaded_file)
    else:
        # Streamlit's UploadedFile and standard file objects support .read()
        uploaded_file.seek(0) if hasattr(uploaded_file, "seek") else None
        file_bytes = uploaded_file.read()

    text = ""

    # 1) Try pdfplumber first
    try:
        text = _extract_with_pdfplumber(file_bytes)
    except Exception as exc:  # noqa: BLE001 - we intentionally catch broadly here
        logger.warning("pdfplumber extraction failed: %s", exc)

    # 2) Fallback to PyPDF2 if pdfplumber produced nothing
    if not text:
        try:
            text = _extract_with_pypdf2(file_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.error("PyPDF2 extraction also failed: %s", exc)

    if not text:
        logger.warning("Could not extract any text from the uploaded PDF.")

    return text


def extract_text_from_path(file_path: str) -> str:
    """
    Convenience wrapper to extract text from a PDF given a file path
    on disk (used for the interview knowledge base / policy folders).

    Args:
        file_path: Absolute or relative path to a .pdf file.

    Returns:
        Extracted text as a string.
    """
    with open(file_path, "rb") as f:
        return extract_text_from_pdf(f.read())
