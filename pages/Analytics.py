"""
Analytics.py
=============
Visual analytics dashboard. Aggregates data across all resumes
screened in the current session and renders charts: match score
distribution, resume category breakdown, suitability breakdown, and
top skills across the candidate pool.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import streamlit as st

from utils.classifier import classify_resume


def render() -> None:
    """Render the Analytics page."""

    st.title("📊 Analytics Dashboard")
    st.caption("Visual insights across all resumes screened in this session.")
    st.markdown("---")

    resumes = st.session_state.resumes
    if not resumes:
        st.info("📄 Upload and screen resumes on the **Resume Screening** page to see analytics here.")
        return

    # ----------------------------------------------------------------
    # Top-level metrics
    # ----------------------------------------------------------------
    rank_results = st.session_state.rank_results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Resumes", len(resumes))
    with col2:
        if rank_results:
            avg_score = sum(r.match_score for r in rank_results) / len(rank_results)
            st.metric("🎯 Avg Match Score", f"{avg_score:.1f}%")
        else:
            st.metric("🎯 Avg Match Score", "—")
    with col3:
        all_skills = [s for data in resumes.values() for s in data["parsed"].skills]
        st.metric("🛠️ Unique Skills Found", len(set(all_skills)))

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    # ----------------------------------------------------------------
    # Chart 1: Match score distribution (bar chart per candidate)
    # ----------------------------------------------------------------
    with chart_col1:
        st.subheader("🎯 Match Score by Candidate")
        if rank_results:
            names = [st.session_state.resumes[r.candidate_name]["parsed"].name for r in rank_results]
            scores = [r.match_score for r in rank_results]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(names, scores, color="#4f46e5")
            ax.set_xlabel("Match Score (%)")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.1f%%", padding=3)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run 'Rank Resumes' on the Resume Screening page to see this chart.")

    # ----------------------------------------------------------------
    # Chart 2: Resume category breakdown (pie chart)
    # ----------------------------------------------------------------
    with chart_col2:
        st.subheader("🏷️ Resume Category Breakdown")
        categories = [classify_resume(data["parsed"].raw_text).category for data in resumes.values()]
        category_counts = Counter(categories)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            category_counts.values(),
            labels=category_counts.keys(),
            autopct="%1.0f%%",
            startangle=90,
            colors=plt.cm.Set2.colors,
        )
        ax.axis("equal")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    chart_col3, chart_col4 = st.columns(2)

    # ----------------------------------------------------------------
    # Chart 3: Top skills across all candidates
    # ----------------------------------------------------------------
    with chart_col3:
        st.subheader("🛠️ Top Skills Across Candidates")
        skill_counter = Counter(all_skills)
        top_skills = skill_counter.most_common(10)

        if top_skills:
            skill_names, skill_freqs = zip(*top_skills)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(skill_names, skill_freqs, color="#0ea5e9")
            ax.invert_yaxis()
            ax.set_xlabel("Number of Candidates")
            ax.bar_label(bars, padding=3)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No skills detected across uploaded resumes yet.")

    # ----------------------------------------------------------------
    # Chart 4: JD match score vs each candidate's own resume (self-check)
    # ----------------------------------------------------------------
    with chart_col4:
        st.subheader("📈 Skill Coverage per Candidate")
        if rank_results:
            names = [st.session_state.resumes[r.candidate_name]["parsed"].name for r in rank_results]
            matching_counts = [len(r.matching_skills) for r in rank_results]
            missing_counts = [len(r.missing_skills) for r in rank_results]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(names, matching_counts, color="#22c55e", label="Matching Skills")
            ax.barh(names, missing_counts, left=matching_counts, color="#ef4444", label="Missing Skills")
            ax.invert_yaxis()
            ax.set_xlabel("Number of Skills")
            ax.legend(loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run 'Rank Resumes' on the Resume Screening page to see this chart.")
