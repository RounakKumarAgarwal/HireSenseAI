"""
classifier.py
=============
Classifies a resume into one of several job categories (Software
Engineer, Data Scientist, Web Developer, AI Engineer, Cloud Engineer,
Cyber Security) using a simple ML pipeline:

    TF-IDF Vectorizer  ->  Multinomial Naive Bayes Classifier

Why Naive Bayes for text classification?
    Naive Bayes is a classic, fast, and surprisingly strong baseline
    for text classification tasks like this one. It trains almost
    instantly even on small synthetic datasets, which makes it ideal
    for a college project that needs a "train on first run" model.

Like ml_model.py, this module trains on a SYNTHETIC dataset built
from category-representative keyword phrases (since we don't have
access to a real labeled resume dataset). This is a standard and
transparent approach for demonstrating an ML classification pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.config import RESUME_CATEGORIES, CLASSIFIER_MODEL_PATH, get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Representative keyword "phrases" per category, used to build a
# synthetic training corpus. Each phrase mimics the kind of language
# that would appear in a real resume for that role.
# ------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, List[str]] = {
    "Software Engineer": [
        "java spring boot microservices rest api backend development",
        "c++ object oriented programming data structures algorithms",
        "software engineer full stack development git agile scrum",
        "python django flask backend api development unit testing",
        "software development lifecycle debugging code review ci cd",
    ],
    "Data Scientist": [
        "python pandas numpy scikit-learn machine learning data analysis",
        "statistics regression classification data visualization matplotlib",
        "data scientist predictive modeling feature engineering jupyter",
        "sql data mining exploratory data analysis hypothesis testing",
        "machine learning models deep learning neural networks pytorch tensorflow",
    ],
    "Web Developer": [
        "html css javascript react frontend web development responsive design",
        "node.js express web development rest api full stack developer",
        "web developer ui ux html5 css3 bootstrap javascript frameworks",
        "wordpress php mysql web development website design",
        "react angular vue frontend framework web application development",
    ],
    "AI Engineer": [
        "deep learning tensorflow pytorch neural networks computer vision nlp",
        "ai engineer machine learning model deployment mlops",
        "natural language processing transformers llm generative ai",
        "computer vision opencv object detection image processing ai",
        "reinforcement learning ai research model training gpu",
    ],
    "Cloud Engineer": [
        "aws azure gcp cloud infrastructure devops terraform",
        "cloud engineer kubernetes docker containerization ci cd pipeline",
        "cloud computing infrastructure as code aws lambda ec2 s3",
        "azure devops cloud migration cloud security networking",
        "google cloud platform kubernetes microservices cloud architecture",
    ],
    "Cyber Security": [
        "cyber security penetration testing vulnerability assessment network security",
        "information security ethical hacking siem incident response",
        "security analyst firewall intrusion detection threat analysis",
        "cyber security compliance risk assessment encryption cryptography",
        "network security malware analysis security operations center soc",
    ],
}


@dataclass
class ClassificationResult:
    """Result of a resume category classification."""
    category: str
    confidence: float
    all_probabilities: dict[str, float]


def _build_training_corpus(samples_per_category: int = 40) -> tuple[List[str], List[str]]:
    """
    Build a synthetic training corpus by combining and shuffling the
    keyword phrases for each category, with light augmentation
    (random phrase combinations) to give the vectorizer more variety.

    Returns:
        Tuple of (texts, labels).
    """
    rng = random.Random(42)
    texts: List[str] = []
    labels: List[str] = []

    for category, phrases in CATEGORY_KEYWORDS.items():
        for _ in range(samples_per_category):
            # Combine 2-3 random phrases from this category to create
            # a longer, more resume-like synthetic training example.
            combo = rng.sample(phrases, k=min(len(phrases), rng.randint(2, 3)))
            texts.append(" ".join(combo))
            labels.append(category)

    return texts, labels


def train_classifier(save_model: bool = True) -> Pipeline:
    """
    Train the TF-IDF + Naive Bayes resume classification pipeline.

    Args:
        save_model: If True, persists the pipeline to
            CLASSIFIER_MODEL_PATH using joblib.

    Returns:
        The trained sklearn Pipeline (vectorizer + classifier).
    """
    texts, labels = _build_training_corpus()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Resume classifier trained. Test accuracy: %.2f%%", accuracy * 100)

    if save_model:
        joblib.dump(pipeline, CLASSIFIER_MODEL_PATH)
        logger.info("Classifier saved to %s", CLASSIFIER_MODEL_PATH)

    return pipeline


def load_classifier() -> Pipeline:
    """
    Load the trained resume classifier pipeline from disk, training
    a new one automatically if none exists yet.

    Returns:
        The loaded (or newly trained) sklearn Pipeline.
    """
    if CLASSIFIER_MODEL_PATH.exists():
        return joblib.load(CLASSIFIER_MODEL_PATH)

    logger.warning("No saved classifier found. Training a new one now...")
    return train_classifier(save_model=True)


def classify_resume(resume_text: str) -> ClassificationResult:
    """
    Classify a resume into one of the predefined job categories.

    Args:
        resume_text: Raw resume text.

    Returns:
        A ClassificationResult with the predicted category,
        confidence score, and full probability breakdown.
    """
    if not resume_text.strip():
        return ClassificationResult(
            category="Unknown", confidence=0.0,
            all_probabilities={cat: 0.0 for cat in RESUME_CATEGORIES},
        )

    pipeline = load_classifier()
    predicted = pipeline.predict([resume_text])[0]
    probabilities = pipeline.predict_proba([resume_text])[0]

    prob_map = dict(zip(pipeline.classes_, probabilities))
    confidence = float(prob_map[predicted])

    logger.info("Classified resume as '%s' with %.1f%% confidence.", predicted, confidence * 100)

    return ClassificationResult(
        category=predicted,
        confidence=round(confidence, 4),
        all_probabilities={k: round(float(v), 4) for k, v in prob_map.items()},
    )
