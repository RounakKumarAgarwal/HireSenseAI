"""
Model_Training.py
===================
Lets HR / the project owner retrain the two ML models used by
HireSense AI directly from the UI:
    1. Suitability Predictor (Random Forest, trained on synthetic
       hiring data)
    2. Resume Classifier (TF-IDF + Naive Bayes, trained on synthetic
       category keyword phrases)

Also shows dataset previews and feature importance so the "black
box" ML pipeline is transparent and explainable - useful for a
college project demo/viva.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.ml_model import (
    generate_sample_dataset,
    train_suitability_model,
    FEATURE_COLUMNS,
)
from utils.classifier import train_classifier, CATEGORY_KEYWORDS
from utils.config import SUITABILITY_MODEL_PATH, CLASSIFIER_MODEL_PATH


def render() -> None:
    """Render the Model Training page."""

    st.title("⚙️ Model Training")
    st.caption("Retrain and inspect the machine learning models that power HireSense AI.")
    st.markdown("---")

    tab_suitability, tab_classifier = st.tabs(
        ["🧪 Suitability Predictor (Random Forest)", "🏷️ Resume Classifier (Naive Bayes)"]
    )

    # ================================================================
    # TAB 1: Suitability Model
    # ================================================================
    with tab_suitability:
        st.subheader("Candidate Suitability Predictor")
        st.markdown(
            "Predicts whether a candidate is **Suitable** or **Not Suitable** for a "
            "role based on match score, experience, matching skills, certifications, "
            "and education level. Trained on a synthetic, heuristic-based hiring dataset."
        )

        model_status = "✅ Trained model found on disk" if SUITABILITY_MODEL_PATH.exists() else "⚠️ No trained model yet"
        st.info(model_status)

        with st.expander("👀 Preview Training Dataset"):
            sample_df = generate_sample_dataset(n_samples=200)
            st.dataframe(sample_df.head(20), use_container_width=True)
            st.caption(f"Full training dataset: {len(sample_df)} synthetic rows (500 used for actual training).")

        if st.button("🔁 Retrain Suitability Model", type="primary"):
            with st.spinner("Training Random Forest model..."):
                model = train_suitability_model(save_model=True)

            st.success("✅ Model retrained and saved successfully!")

            # Feature importance chart
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.barh(FEATURE_COLUMNS, importances, color="#4f46e5")
            ax.set_xlabel("Importance")
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.2f", padding=3)
            fig.tight_layout()
            st.pyplot(fig)

    # ================================================================
    # TAB 2: Resume Classifier
    # ================================================================
    with tab_classifier:
        st.subheader("Resume Category Classifier")
        st.markdown(
            "Classifies resumes into job categories using a TF-IDF + Naive Bayes "
            "pipeline, trained on synthetic keyword phrases representative of each category."
        )

        model_status = "✅ Trained model found on disk" if CLASSIFIER_MODEL_PATH.exists() else "⚠️ No trained model yet"
        st.info(model_status)

        with st.expander("👀 Preview Category Keywords Used for Training"):
            for category, phrases in CATEGORY_KEYWORDS.items():
                st.markdown(f"**{category}**")
                for phrase in phrases:
                    st.caption(f"• {phrase}")

        if st.button("🔁 Retrain Resume Classifier", type="primary"):
            with st.spinner("Training TF-IDF + Naive Bayes classifier..."):
                train_classifier(save_model=True)
            st.success("✅ Classifier retrained and saved successfully!")

    st.markdown("---")
    st.caption(
        "ℹ️ Both models are trained automatically on first use if no saved model "
        "is found, so manual retraining here is optional - useful mainly for "
        "demonstrating the ML pipeline during a project viva/demo."
    )
