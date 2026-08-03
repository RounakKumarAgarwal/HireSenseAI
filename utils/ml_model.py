"""
ml_model.py
===========
Trains and serves a Random Forest classifier that predicts whether a
candidate is "Suitable" or "Not Suitable" for a role, along with a
probability/confidence score.

Since we don't have access to a real company hiring dataset, this
module generates a realistic SYNTHETIC dataset based on sensible
hiring heuristics (more matching skills + more experience + higher
JD match score => higher chance of being "Suitable"). This is a
common and acceptable approach for a college ML project - it lets us
demonstrate the full train -> save -> load -> predict pipeline.

Features used:
    1. match_score        - Resume vs JD TF-IDF match % (0-100)
    2. years_experience    - Estimated years of experience
    3. num_matching_skills - Count of skills that match the JD
    4. num_certifications  - Count of certifications listed
    5. education_score     - Encoded education level (0-3)

Target:
    suitable (1 = Suitable, 0 = Not Suitable)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils.config import SUITABILITY_MODEL_PATH, get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "match_score",
    "years_experience",
    "num_matching_skills",
    "num_certifications",
    "education_score",
]

EDUCATION_LEVELS = {
    "high school": 0,
    "diploma": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 3,
}


@dataclass
class SuitabilityResult:
    """Result of a candidate suitability prediction."""
    prediction: str          # "Suitable" or "Not Suitable"
    probability: float       # Confidence of the predicted class (0-1)


def generate_sample_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic hiring dataset for training the suitability
    model. Suitability is determined by a weighted, noisy combination
    of the input features, so the model has real signal to learn
    from (rather than pure randomness).

    Args:
        n_samples: Number of synthetic candidate rows to generate.
        random_state: Seed for reproducibility.

    Returns:
        A pandas DataFrame with feature columns + a "suitable" label.
    """
    rng = np.random.default_rng(random_state)

    match_score = rng.uniform(0, 100, n_samples)
    years_experience = rng.uniform(0, 15, n_samples)
    num_matching_skills = rng.integers(0, 15, n_samples)
    num_certifications = rng.integers(0, 6, n_samples)
    education_score = rng.integers(0, 4, n_samples)

    # Weighted "hiring score" formula with a bit of random noise
    hiring_score = (
        0.4 * (match_score / 100)
        + 0.25 * (years_experience / 15)
        + 0.2 * (num_matching_skills / 15)
        + 0.1 * (num_certifications / 6)
        + 0.05 * (education_score / 3)
        + rng.normal(0, 0.08, n_samples)  # noise for realism
    )

    suitable = (hiring_score > 0.55).astype(int)

    df = pd.DataFrame(
        {
            "match_score": match_score,
            "years_experience": years_experience,
            "num_matching_skills": num_matching_skills,
            "num_certifications": num_certifications,
            "education_score": education_score,
            "suitable": suitable,
        }
    )

    logger.info(
        "Generated synthetic dataset: %d rows, %d suitable / %d not suitable.",
        n_samples, suitable.sum(), n_samples - suitable.sum(),
    )
    return df


def train_suitability_model(save_model: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the synthetic hiring dataset
    and optionally persist it to disk with joblib.

    Args:
        save_model: If True, saves the trained model to
            SUITABILITY_MODEL_PATH (defined in config.py).

    Returns:
        The trained RandomForestClassifier instance.
    """
    df = generate_sample_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["suitable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Suitability model trained. Test accuracy: %.2f%%", accuracy * 100)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"]))

    if save_model:
        joblib.dump(model, SUITABILITY_MODEL_PATH)
        logger.info("Model saved to %s", SUITABILITY_MODEL_PATH)

    return model


def load_suitability_model() -> RandomForestClassifier:
    """
    Load the trained suitability model from disk. If no saved model
    exists yet, trains one automatically first.

    Returns:
        The loaded (or newly trained) RandomForestClassifier.
    """
    if SUITABILITY_MODEL_PATH.exists():
        return joblib.load(SUITABILITY_MODEL_PATH)

    logger.warning("No saved suitability model found. Training a new one now...")
    return train_suitability_model(save_model=True)


def predict_suitability(
    match_score: float,
    years_experience: float,
    num_matching_skills: int,
    num_certifications: int,
    education_level: str = "bachelor",
) -> SuitabilityResult:
    """
    Predict whether a candidate is suitable for a role.

    Args:
        match_score: Resume-vs-JD TF-IDF match percentage (0-100).
        years_experience: Estimated years of professional experience.
        num_matching_skills: Number of skills matching the JD.
        num_certifications: Number of certifications listed.
        education_level: One of "high school", "diploma", "bachelor",
            "master", "phd".

    Returns:
        A SuitabilityResult with the prediction label and confidence.
    """
    model = load_suitability_model()
    education_score = EDUCATION_LEVELS.get(education_level.lower(), 2)

    features = pd.DataFrame(
        [[match_score, years_experience, num_matching_skills, num_certifications, education_score]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])

    label = "Suitable" if prediction == 1 else "Not Suitable"
    logger.info("Predicted '%s' with %.1f%% confidence.", label, confidence * 100)

    return SuitabilityResult(prediction=label, probability=round(confidence, 4))
