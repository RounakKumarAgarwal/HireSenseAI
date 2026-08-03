"""
Interview_Generator.py
========================
The AI Interview Question Generator page. HR selects a candidate
(and optionally uses the job description already provided on the
Resume Screening page), and Groq generates tailored Technical, HR,
Behavioral, and Coding interview questions.
"""

from __future__ import annotations

import streamlit as st

from utils.interview_generator import generate_interview_questions


def render() -> None:
    """Render the Interview Generator page."""

    st.title("🎤 AI Interview Question Generator")
    st.caption("Generate tailored interview questions based on a candidate's resume, skills, and the job description.")
    st.markdown("---")

    if not st.session_state.resumes:
        st.info("📄 Upload resumes on the **Resume Screening** page first to generate interview questions.")
        return

    candidate_filenames = list(st.session_state.resumes.keys())
    selected_file = st.selectbox(
        "Select a candidate",
        candidate_filenames,
        format_func=lambda f: f"{st.session_state.resumes[f]['parsed'].name} ({f})",
    )

    data = st.session_state.resumes[selected_file]
    parsed = data["parsed"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**👤 Candidate:** {parsed.name}")
        st.markdown(f"**🛠️ Skills:** {', '.join(parsed.skills) or 'None detected'}")
    with col2:
        num_questions = st.slider("Questions per category", min_value=3, max_value=8, value=5)

    use_jd = st.checkbox(
        "Include the job description as context (from Resume Screening page)",
        value=bool(st.session_state.jd_text),
        disabled=not bool(st.session_state.jd_text),
    )

    if st.button("🎯 Generate Interview Questions", type="primary"):
        jd_context = st.session_state.jd_text if use_jd else ""
        with st.spinner("Groq is crafting tailored interview questions..."):
            questions = generate_interview_questions(
                resume_text=data["text"],
                skills=parsed.skills,
                job_description=jd_context,
                num_per_category=num_questions,
            )
        st.session_state.interview_questions = questions

    questions = st.session_state.interview_questions
    if questions and (questions.technical or questions.hr or questions.behavioral or questions.coding):
        st.markdown("---")
        tab_tech, tab_hr, tab_behavioral, tab_coding = st.tabs(
            ["💻 Technical", "🧑‍💼 HR", "🧭 Behavioral", "⌨️ Coding"]
        )

        with tab_tech:
            _render_question_list(questions.technical, "No technical questions generated.")
        with tab_hr:
            _render_question_list(questions.hr, "No HR questions generated.")
        with tab_behavioral:
            _render_question_list(questions.behavioral, "No behavioral questions generated.")
        with tab_coding:
            _render_question_list(questions.coding, "No coding questions generated.")


def _render_question_list(questions: list[str], empty_message: str) -> None:
    """Render a numbered list of interview questions inside a styled card."""
    if not questions:
        st.info(empty_message)
        return

    for idx, question in enumerate(questions, start=1):
        st.markdown(
            f'<div class="hs-card">❓ <strong>Q{idx}.</strong> {question}</div>',
            unsafe_allow_html=True,
        )
