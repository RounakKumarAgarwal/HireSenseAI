"""
skill_extractor.py
===================
Detects technical skills mentioned inside resume / job-description
text, using a simple but effective keyword-matching approach.

Why keyword matching instead of an ML model?
    For a curated, known list of technical skills, exact/fuzzy
    keyword matching is fast, transparent, and 100% explainable -
    perfect for a college project where you need to be able to
    explain exactly *why* a skill was detected. Extending the skill
    list is as easy as adding a string to `KNOWN_SKILLS` in
    config.py - no retraining required.
"""

from __future__ import annotations

import re
from typing import List

from utils.config import KNOWN_SKILLS, get_logger

logger = get_logger(__name__)


def _build_skill_pattern(skill: str) -> re.Pattern:
    """
    Build a case-insensitive regex pattern that matches a skill as a
    whole word/phrase (so "R" doesn't match inside "React", and
    "Java" doesn't match inside "JavaScript").
    """
    # Escape special regex characters (e.g. "C++", "C#") then wrap
    # with word boundaries where safe to do so.
    escaped = re.escape(skill)
    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


# Pre-compile all skill patterns once at import time for performance.
_SKILL_PATTERNS: dict[str, re.Pattern] = {
    skill: _build_skill_pattern(skill) for skill in KNOWN_SKILLS
}


def extract_skills(text: str, custom_skills: List[str] | None = None) -> List[str]:
    """
    Extract all known technical skills mentioned in the given text.

    Args:
        text: Raw resume or job-description text.
        custom_skills: Optional extra list of skills to search for,
            in addition to the built-in KNOWN_SKILLS list. Useful if
            a specific job description mentions a niche tool.

    Returns:
        A sorted list of unique skill names found in the text
        (preserves the canonical casing from KNOWN_SKILLS, e.g.
        "javascript" in the resume -> "JavaScript" in the result).
    """
    if not text:
        return []

    found: set[str] = set()

    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
            found.add(skill)

    if custom_skills:
        for skill in custom_skills:
            if _build_skill_pattern(skill).search(text):
                found.add(skill)

    result = sorted(found)
    logger.info("Extracted %d skills from text.", len(result))
    return result


def compare_skills(resume_skills: List[str], jd_skills: List[str]) -> dict:
    """
    Compare the skills found in a resume against the skills required
    by a job description.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.

    Returns:
        A dictionary with:
            - "matching": skills present in both
            - "missing": skills required by JD but absent from resume
            - "extra": skills in resume not mentioned in JD
    """
    resume_set = {s.lower() for s in resume_skills}
    jd_set = {s.lower() for s in jd_skills}

    # Map lowercase back to canonical casing for display purposes
    canonical = {s.lower(): s for s in set(resume_skills) | set(jd_skills)}

    matching = sorted(canonical[s] for s in (resume_set & jd_set))
    missing = sorted(canonical[s] for s in (jd_set - resume_set))
    extra = sorted(canonical[s] for s in (resume_set - jd_set))

    return {"matching": matching, "missing": missing, "extra": extra}
