"""
resume_parser.py
=================
Extracts structured fields (name, email, phone, education, experience,
certifications, skills) from raw resume text using regex + heuristic
rules.

Why regex/heuristics instead of a heavy NLP model?
    Resumes are semi-structured documents (they use predictable
    patterns like emails, phone numbers, and section headings such
    as "EDUCATION" or "EXPERIENCE"). Regex-based extraction is fast,
    dependency-free, and transparent - ideal for a college project
    where explainability matters. This module can later be swapped
    for a spaCy NER model without changing its public interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from utils.skill_extractor import extract_skills
from utils.config import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Regex patterns for contact details
# ------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(
    r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})"
)

# Section headings we look for when splitting a resume into blocks.
SECTION_HEADINGS = {
    "education": ["education", "academic background", "qualifications"],
    "experience": ["experience", "work experience", "employment history",
                   "professional experience"],
    "certifications": ["certifications", "certificates", "licenses"],
    "skills": ["skills", "technical skills", "core competencies"],
    "projects": ["projects", "personal projects", "academic projects"],
}


@dataclass
class ParsedResume:
    """Structured representation of a parsed resume."""
    name: str = "Not Found"
    email: str = "Not Found"
    phone: str = "Not Found"
    skills: List[str] = field(default_factory=list)
    education: str = "Not Found"
    experience: str = "Not Found"
    certifications: str = "Not Found"
    raw_text: str = ""


def _extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else "Not Found"


def _extract_phone(text: str) -> str:
    for match in PHONE_PATTERN.finditer(text):
        candidate = match.group(0)
        digits_only = re.sub(r"\D", "", candidate)
        # A real phone number typically has 7-13 digits.
        if 7 <= len(digits_only) <= 13:
            return candidate.strip()
    return "Not Found"


def _extract_name(text: str) -> str:
    """
    Heuristic: the candidate's name is usually the first non-empty
    line of the resume, and it typically doesn't contain digits,
    '@', or common resume keywords.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocklist = {"resume", "curriculum vitae", "cv"}

    for line in lines[:5]:  # only check the first few lines
        lower = line.lower()
        if lower in blocklist:
            continue
        if EMAIL_PATTERN.search(line) or PHONE_PATTERN.search(line):
            continue
        if any(char.isdigit() for char in line):
            continue
        word_count = len(line.split())
        if 1 <= word_count <= 5:
            return line.title()

    return "Not Found"


def _extract_section(text: str, heading_variants: List[str]) -> str:
    """
    Extract the block of text that follows one of the given section
    headings, stopping at the next recognised heading.

    Args:
        text: Full resume text.
        heading_variants: Alternative spellings for the heading we
            want (e.g. ["education", "academic background"]).

    Returns:
        The extracted section text, or "Not Found" if no matching
        heading exists.
    """
    lines = text.splitlines()
    all_headings = [h for variants in SECTION_HEADINGS.values() for h in variants]

    start_idx = None
    for idx, line in enumerate(lines):
        clean = line.strip().lower().strip(":")
        if clean in heading_variants:
            start_idx = idx + 1
            break

    if start_idx is None:
        return "Not Found"

    collected: List[str] = []
    for line in lines[start_idx:]:
        clean = line.strip().lower().strip(":")
        if clean in all_headings and clean not in heading_variants:
            break  # reached the next section
        if line.strip():
            collected.append(line.strip())

    section_text = "\n".join(collected).strip()
    return section_text if section_text else "Not Found"


def parse_resume(text: str) -> ParsedResume:
    """
    Parse raw resume text into a structured ParsedResume object.

    Args:
        text: Raw text extracted from a resume PDF.

    Returns:
        A ParsedResume dataclass instance with all fields populated.
    """
    if not text:
        logger.warning("Empty text passed to parse_resume().")
        return ParsedResume()

    parsed = ParsedResume(
        name=_extract_name(text),
        email=_extract_email(text),
        phone=_extract_phone(text),
        skills=extract_skills(text),
        education=_extract_section(text, SECTION_HEADINGS["education"]),
        experience=_extract_section(text, SECTION_HEADINGS["experience"]),
        certifications=_extract_section(text, SECTION_HEADINGS["certifications"]),
        raw_text=text,
    )

    logger.info("Parsed resume for candidate: %s", parsed.name)
    return parsed


def estimate_years_of_experience(experience_text: str) -> float:
    """
    Best-effort estimate of total years of experience by scanning
    for date ranges like "2019 - 2023" or "2020 - Present" inside
    the experience section.

    Args:
        experience_text: The extracted "experience" section text.

    Returns:
        Estimated total years of experience (float, rounded to 1
        decimal place). Returns 0.0 if no date ranges are found.
    """
    if not experience_text or experience_text == "Not Found":
        return 0.0

    year_pattern = re.compile(r"(19|20)\d{2}")
    present_pattern = re.compile(r"present|current", re.IGNORECASE)

    total_years = 0.0
    lines = experience_text.split("\n")

    for line in lines:
        years = year_pattern.findall(line)
        full_matches = year_pattern.finditer(line)
        year_values = [int(m.group(0)) for m in full_matches]

        if len(year_values) >= 2:
            total_years += abs(year_values[-1] - year_values[0])
        elif len(year_values) == 1 and present_pattern.search(line):
            import datetime
            current_year = datetime.datetime.now().year
            total_years += abs(current_year - year_values[0])

    return round(total_years, 1)
