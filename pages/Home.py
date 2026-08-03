"""
Home.py
=======
The landing dashboard for HireSense AI. Gives HR a quick overview of
the system's capabilities and a snapshot of current session activity
(resumes loaded, average match score, top candidate, etc).
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Home dashboard page."""

    st.title("🧠 HireSense AI")
    st.markdown(
        "#### Intelligent Recruitment & Candidate Evaluation System "
        "using Machine Learning and Retrieval-Augmented Generation (RAG)"
    )
    st.markdown("---")

    # --------------------------------------------------------------
    # Quick stats row
    # --------------------------------------------------------------
    resumes = st.session_state.resumes
    rank_results = st.session_state.rank_results

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Resumes Loaded", len(resumes))

    with col2:
        if rank_results:
            avg_score = sum(r.match_score for r in rank_results) / len(rank_results)
            st.metric("🎯 Avg Match Score", f"{avg_score:.1f}%")
        else:
            st.metric("🎯 Avg Match Score", "—")

    with col3:
        if rank_results:
            top_candidate = rank_results[0].candidate_name
            st.metric("🏆 Top Candidate", top_candidate)
        else:
            st.metric("🏆 Top Candidate", "—")

    with col4:
        jd_status = "✅ Loaded" if st.session_state.jd_text else "❌ Not Set"
        st.metric("📋 Job Description", jd_status)

    st.markdown("---")

    # --------------------------------------------------------------
    # Feature overview cards
    # --------------------------------------------------------------
    st.subheader("✨ What HireSense AI Can Do")

    features = [
        ("📄", "Resume Screening", "Upload resumes, auto-parse candidate details, and rank against a job description using TF-IDF matching."),
        ("🧪", "Suitability Prediction", "A Random Forest ML model predicts Suitable / Not Suitable with a confidence score."),
        ("🏷️", "Resume Classification", "Automatically categorize resumes into roles like Data Scientist, Web Developer, or AI Engineer."),
        ("✍️", "AI Resume Summaries", "Groq-powered LLM generates a concise, professional summary of every candidate."),
        ("💬", "Resume Chatbot (RAG)", "Ask natural-language questions about any candidate's resume - answers are grounded only in that resume."),
        ("📜", "Policy Chatbot (RAG)", "Upload company policy PDFs and get instant, grounded answers to HR policy questions."),
        ("🎤", "Interview Question Generator", "Generate tailored Technical, HR, Behavioral, and Coding interview questions."),
        ("📚", "Interview Knowledge Base", "A RAG chatbot over your library of interview-prep guides (Java, Python, SQL, ML, etc)."),
    ]

    cols = st.columns(2)
    for idx, (emoji, title, desc) in enumerate(features):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="hs-card">
                    <h4>{emoji} {title}</h4>
                    <p style="color:#4b5563; margin-bottom:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --------------------------------------------------------------
    # Getting started guide
    # --------------------------------------------------------------
    with st.expander("🚀 Getting Started", expanded=not resumes):
        st.markdown(
            """
            1. Go to **📄 Resume Screening** and upload one or more candidate resumes (PDF).
            2. Paste or upload a **Job Description** on the same page.
            3. Click **Rank Resumes** to see match scores, suitability predictions, and category classification.
            4. Head to **💬 Resume Chat** to ask questions about any specific candidate.
            5. Upload your HR policies in **📜 Policy Chat** to enable the policy chatbot.
            6. Use **🎤 Interview Generator** to create tailored interview questions for your top candidates.
            7. Check **📊 Analytics** for visual insights across all screened candidates.
            8. Visit **⚙️ Model Training** if you want to retrain the ML models from scratch.
            """
        )
