"""
matcher.py
==========
Computes a "Resume Match Score" between a candidate's resume and a
job description (JD), using classic TF-IDF + Cosine Similarity.

Why TF-IDF + Cosine Similarity instead of embeddings here?
    This module powers the fast, real-time "Resume Ranking" table
    where dozens of resumes may need to be scored against one JD.
    TF-IDF is extremely fast, requires no GPU/model download, and is
    a well-understood, explainable baseline that's perfect for a
    college project. (The RAG chatbot modules use embeddings instead,
    where semantic understanding matters more than speed.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.skill_extractor import extract_skills, compare_skills
from utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Structured result of a resume-vs-JD comparison."""
    candidate_name: str
    match_score: float  # 0-100 percentage
    matching_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute the cosine similarity between a resume and a job
    description using TF-IDF vectorization.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.

    Returns:
        A match score between 0.0 and 100.0 (percentage).
    """
    if not resume_text or not jd_text:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    except ValueError:
        # Happens if both texts are empty after stopword removal
        logger.warning("TF-IDF vectorization produced an empty vocabulary.")
        return 0.0

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score = round(float(similarity) * 100, 2)
    return score


def rank_resumes(resumes: dict[str, str], jd_text: str) -> List[MatchResult]:
    """
    Rank multiple resumes against a single job description.

    Args:
        resumes: Dictionary mapping {candidate_name: resume_text}.
        jd_text: Raw job description text.

    Returns:
        A list of MatchResult objects, sorted by match_score
        descending (best match first).
    """
    jd_skills = extract_skills(jd_text)
    results: List[MatchResult] = []

    for candidate_name, resume_text in resumes.items():
        score = compute_match_score(resume_text, jd_text)
        resume_skills = extract_skills(resume_text)
        skill_comparison = compare_skills(resume_skills, jd_skills)

        results.append(
            MatchResult(
                candidate_name=candidate_name,
                match_score=score,
                matching_skills=skill_comparison["matching"],
                missing_skills=skill_comparison["missing"],
                extra_skills=skill_comparison["extra"],
            )
        )

    results.sort(key=lambda r: r.match_score, reverse=True)
    logger.info("Ranked %d resumes against the job description.", len(results))
    return results
