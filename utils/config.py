"""
config.py
=========
Central configuration module for HireSense AI.

This file is the single source of truth for:
    - File system paths (data folders, model folders)
    - Environment variable loading (Groq API key, model name)
    - Global constants (skill list, resume categories, etc.)
    - Logging setup

Why this module exists:
    Instead of scattering file paths and constants across every
    module, we define them ONCE here and import them everywhere
    else. This makes the project easier to maintain - if a folder
    path changes, we only change it in one place.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment variables from .env file (if present)
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# Base project directory (the folder that contains app.py)
# ------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Data / storage directories
# ------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
RESUME_DIR: Path = DATA_DIR / "resumes"
POLICY_DIR: Path = DATA_DIR / "policies"
INTERVIEW_KB_DIR: Path = DATA_DIR / "interview_questions"
MODELS_DIR: Path = BASE_DIR / "models"
ASSETS_DIR: Path = BASE_DIR / "assets"

# Make sure all required directories exist at import time.
for _directory in (RESUME_DIR, POLICY_DIR, INTERVIEW_KB_DIR, MODELS_DIR, ASSETS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Saved model file paths
# ------------------------------------------------------------------
SUITABILITY_MODEL_PATH: Path = MODELS_DIR / "suitability_model.joblib"
CLASSIFIER_MODEL_PATH: Path = MODELS_DIR / "resume_classifier.joblib"
CLASSIFIER_VECTORIZER_PATH: Path = MODELS_DIR / "classifier_vectorizer.joblib"

# FAISS index + metadata cache paths (created dynamically at runtime,
# but we keep a default cache folder for convenience)
FAISS_CACHE_DIR: Path = MODELS_DIR / "faiss_cache"
FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Groq LLM configuration
# ------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ------------------------------------------------------------------
# Embedding model configuration (SentenceTransformers)
# ------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ------------------------------------------------------------------
# Skill dictionary used by skill_extractor.py
# Feel free to append more skills here - the extractor picks these
# up automatically, no other code changes required.
# ------------------------------------------------------------------
KNOWN_SKILLS: list[str] = [
    "Python", "Java", "SQL", "C++", "C#", "HTML", "CSS", "JavaScript",
    "TypeScript", "React", "Angular", "Vue", "Node.js", "Express",
    "Spring Boot", "Django", "Flask", "FastAPI",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "Keras", "NLP", "Computer Vision", "OpenCV",
    "Docker", "Kubernetes", "Jenkins", "CI/CD",
    "AWS", "Azure", "GCP", "Google Cloud",
    "Git", "GitHub", "GitLab", "Linux", "Bash",
    "Power BI", "Tableau", "Excel",
    "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn",
    "MongoDB", "MySQL", "PostgreSQL", "Redis", "Firebase",
    "Hadoop", "Spark", "Kafka",
    "REST API", "GraphQL", "Microservices",
    "Agile", "Scrum", "Jira",
]

# ------------------------------------------------------------------
# Resume classification categories
# ------------------------------------------------------------------
RESUME_CATEGORIES: list[str] = [
    "Software Engineer",
    "Data Scientist",
    "Web Developer",
    "AI Engineer",
    "Cloud Engineer",
    "Cyber Security",
]

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger instance.

    Every module in the project should call this instead of using
    print statements, so that we get consistent, timestamped,
    leveled log output across the whole app.

    Args:
        name: Usually pass __name__ from the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
