"""
summarizer.py
=============
Generates a concise, professional 4-5 sentence summary of a
candidate's resume using the Groq LLM API.

This module also defines the shared Groq client helper functions
(`get_groq_client` and `generate_completion`) that other LLM-powered
modules in the project (interview_generator.py, rag_resume.py,
rag_policy.py, rag_interview.py) import and reuse - this avoids
duplicating API-call boilerplate across the codebase.
"""

from __future__ import annotations

from groq import Groq

from utils.config import GROQ_API_KEY, GROQ_MODEL, get_logger

logger = get_logger(__name__)

_client: Groq | None = None


def get_groq_client() -> Groq:
    """
    Lazily create (and cache) a single Groq API client instance for
    the whole app, so we don't re-authenticate on every call.

    Returns:
        A configured Groq client.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 700,
) -> str:
    """
    Send a single-turn chat completion request to the Groq LLM.

    Args:
        system_prompt: Instructions that define the LLM's role/behavior.
        user_prompt: The actual task/question for the LLM.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The LLM's text response, stripped of leading/trailing whitespace.
        Returns a friendly error message string if the API call fails,
        so the Streamlit UI never crashes on an LLM error.
    """
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Groq API call failed: %s", exc)
        return (
            "⚠️ Could not reach the Groq LLM API. Please check that "
            "GROQ_API_KEY is set correctly in your .env file, and that "
            "you have an active internet connection. "
            f"(Error: {exc})"
        )


def summarize_resume(resume_text: str, candidate_name: str = "the candidate") -> str:
    """
    Generate a professional 4-5 sentence summary of a resume.

    Args:
        resume_text: Raw resume text.
        candidate_name: Candidate's name, used to personalize the prompt.

    Returns:
        A 4-5 sentence professional summary as plain text.
    """
    if not resume_text.strip():
        return "No resume text available to summarize."

    # Truncate very long resumes to keep the prompt within token limits
    truncated_text = resume_text[:6000]

    system_prompt = (
        "You are an expert HR recruiter and resume analyst. You write "
        "concise, professional, and factual candidate summaries used "
        "by hiring managers to quickly evaluate applicants. Never "
        "invent information that isn't in the resume."
    )

    user_prompt = (
        f"Read the resume below for {candidate_name} and write a "
        f"professional summary in EXACTLY 4-5 sentences. Cover their "
        f"key skills, most relevant experience, and overall strengths "
        f"as a candidate. Do not use bullet points - write flowing "
        f"prose.\n\n"
        f"RESUME:\n{truncated_text}"
    )

    summary = generate_completion(system_prompt, user_prompt, temperature=0.4, max_tokens=350)
    logger.info("Generated summary for %s.", candidate_name)
    return summary
