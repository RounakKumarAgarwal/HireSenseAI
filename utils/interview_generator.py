"""
interview_generator.py
=======================
Generates tailored interview questions using the Groq LLM, based on
a candidate's resume, extracted skills, and (optionally) the job
description they're being considered for.

Four question categories are generated:
    - Technical questions   (role/skill-specific knowledge)
    - HR questions            (culture fit, career goals, logistics)
    - Behavioral questions    (past experiences, soft skills, STAR-style)
    - Coding questions        (hands-on programming problems)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from utils.summarizer import generate_completion
from utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class InterviewQuestions:
    """Structured container for all generated interview questions."""
    technical: List[str] = field(default_factory=list)
    hr: List[str] = field(default_factory=list)
    behavioral: List[str] = field(default_factory=list)
    coding: List[str] = field(default_factory=list)


def _parse_json_response(raw_response: str) -> dict:
    """
    Safely parse a JSON object out of the LLM's raw text response,
    even if it's wrapped in markdown code fences or has extra text
    around it.

    Args:
        raw_response: The raw text returned by the LLM.

    Returns:
        A parsed dictionary. Returns an empty dict if parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json|```", "", raw_response).strip()

    # Try to isolate the outermost { ... } block just in case there's
    # leading/trailing commentary from the model.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    json_str = match.group(0) if match else cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON response: %s", exc)
        return {}


def generate_interview_questions(
    resume_text: str,
    skills: List[str],
    job_description: str = "",
    num_per_category: int = 5,
) -> InterviewQuestions:
    """
    Generate a full set of interview questions tailored to a
    candidate's resume, skills, and (optionally) a job description.

    Args:
        resume_text: Raw resume text.
        skills: List of extracted technical skills.
        job_description: Optional job description text for extra context.
        num_per_category: How many questions to generate per category.

    Returns:
        An InterviewQuestions dataclass with all four categories filled.
    """
    truncated_resume = resume_text[:4000]
    skills_str = ", ".join(skills) if skills else "General technical skills"
    jd_context = f"\n\nJOB DESCRIPTION:\n{job_description[:2000]}" if job_description else ""

    system_prompt = (
        "You are an experienced technical interviewer and HR panel "
        "member. You design high-quality, role-relevant interview "
        "questions. You always respond with STRICT, VALID JSON and "
        "nothing else - no markdown, no commentary, no code fences."
    )

    user_prompt = (
        f"Based on the candidate resume and skills below, generate "
        f"exactly {num_per_category} interview questions for EACH of "
        f"these 4 categories: technical, hr, behavioral, coding.\n\n"
        f"CANDIDATE SKILLS: {skills_str}\n"
        f"RESUME:\n{truncated_resume}"
        f"{jd_context}\n\n"
        f"Respond ONLY with valid JSON in exactly this format:\n"
        f'{{"technical": ["question1", "question2", ...], '
        f'"hr": ["question1", ...], '
        f'"behavioral": ["question1", ...], '
        f'"coding": ["question1", ...]}}'
    )

    raw_response = generate_completion(system_prompt, user_prompt, temperature=0.6, max_tokens=1500)
    parsed = _parse_json_response(raw_response)

    if not parsed:
        logger.warning("Falling back to empty interview question set due to parse failure.")
        return InterviewQuestions()

    questions = InterviewQuestions(
        technical=parsed.get("technical", [])[:num_per_category],
        hr=parsed.get("hr", [])[:num_per_category],
        behavioral=parsed.get("behavioral", [])[:num_per_category],
        coding=parsed.get("coding", [])[:num_per_category],
    )

    logger.info(
        "Generated interview questions: %d technical, %d hr, %d behavioral, %d coding.",
        len(questions.technical), len(questions.hr),
        len(questions.behavioral), len(questions.coding),
    )
    return questions
