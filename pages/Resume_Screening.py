"""
Resume_Screening.py
====================
The core screening workflow page. Lets HR:
    1. Upload one or more resume PDFs (extract + parse + display)
    2. Provide a job description (paste or upload PDF)
    3. Rank resumes against the JD using TF-IDF match scoring
    4. See ML-predicted suitability (Random Forest) per candidate
    5. See auto-classified job category per candidate
    6. Generate an AI (Groq) professional summary per candidate
"""

from __future__ import annotations

import streamlit as st

from utils.pdf_parser import extract_text_from_pdf
from utils.resume_parser import parse_resume, estimate_years_of_experience
from utils.skill_extractor import extract_skills
from utils.matcher import rank_resumes
from utils.ml_model import predict_suitability
from utils.classifier import classify_resume
from utils.summarizer import summarize_resume
from utils.config import get_logger

logger = get_logger(__name__)


def _process_uploaded_resumes(uploaded_files) -> None:
    """Extract text and parse each newly uploaded resume PDF, storing
    the results in st.session_state.resumes."""
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.resumes:
            continue  # already processed this session

        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            parsed = parse_resume(text)

        st.session_state.resumes[uploaded_file.name] = {
            "text": text,
            "parsed": parsed,
            "rag_engine": None,  # built lazily on the Resume Chat page
            "summary": None,     # generated on demand
        }
        logger.info("Processed uploaded resume: %s", uploaded_file.name)


def _count_certifications(cert_text: str) -> int:
    """Rough count of certifications based on newline-separated entries."""
    if not cert_text or cert_text == "Not Found":
        return 0
    return len([line for line in cert_text.split("\n") if line.strip()])


def render() -> None:
    """Render the Resume Screening page."""

    st.title("📄 Resume Screening")
    st.caption("Upload resumes, provide a job description, and let HireSense AI do the rest.")
    st.markdown("---")

    tab_upload, tab_jd, tab_rank, tab_details = st.tabs(
        ["📤 Upload Resumes", "📋 Job Description", "🏆 Ranking", "🔍 Candidate Details"]
    )

    # ================================================================
    # TAB 1: Upload Resumes
    # ================================================================
    with tab_upload:
        st.subheader("Upload Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            _process_uploaded_resumes(uploaded_files)

        if st.session_state.resumes:
            st.success(f"✅ {len(st.session_state.resumes)} resume(s) loaded this session.")

            for filename, data in st.session_state.resumes.items():
                parsed = data["parsed"]
                with st.expander(f"👤 {parsed.name}  —  {filename}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**📧 Email:** {parsed.email}")
                        st.markdown(f"**📞 Phone:** {parsed.phone}")
                        st.markdown(f"**🎓 Education:**\n\n{parsed.education}")
                    with col2:
                        st.markdown(f"**💼 Experience:**\n\n{parsed.experience}")
                        st.markdown(f"**📜 Certifications:**\n\n{parsed.certifications}")

                    st.markdown("**🛠️ Skills Detected:**")
                    if parsed.skills:
                        st.write(", ".join(f"`{s}`" for s in parsed.skills))
                    else:
                        st.write("No known skills detected.")

                    show_raw = st.checkbox(
                        "📃 Show Raw Extracted Text", key=f"show_raw_{filename}"
                    )
                    if show_raw:
                        st.text_area(
                            "Raw text", data["text"], height=200,
                            key=f"raw_{filename}", label_visibility="collapsed",
                        )

            if st.button("🗑️ Clear All Resumes"):
                st.session_state.resumes = {}
                st.session_state.rank_results = []
                st.rerun()
        else:
            st.info("No resumes uploaded yet. Upload PDF resumes above to get started.")

    # ================================================================
    # TAB 2: Job Description
    # ================================================================
    with tab_jd:
        st.subheader("Provide the Job Description")
        jd_input_method = st.radio(
            "How would you like to provide the JD?",
            ["✍️ Paste Text", "📄 Upload PDF"],
            horizontal=True,
        )

        if jd_input_method == "✍️ Paste Text":
            jd_text = st.text_area(
                "Paste the job description here",
                value=st.session_state.jd_text,
                height=250,
                placeholder="e.g. We are looking for a Python developer with 3+ years experience in Django, REST APIs, and AWS...",
            )
            if jd_text != st.session_state.jd_text:
                st.session_state.jd_text = jd_text
        else:
            jd_file = st.file_uploader("Upload job description PDF", type=["pdf"], key="jd_pdf")
            if jd_file is not None:
                with st.spinner("Extracting job description text..."):
                    st.session_state.jd_text = extract_text_from_pdf(jd_file)
                st.text_area("Extracted JD text", st.session_state.jd_text, height=250, disabled=True)

        if st.session_state.jd_text:
            jd_skills = extract_skills(st.session_state.jd_text)
            st.markdown("**🛠️ Skills Detected in JD:**")
            st.write(", ".join(f"`{s}`" for s in jd_skills) if jd_skills else "No known skills detected.")

    # ================================================================
    # TAB 3: Ranking
    # ================================================================
    with tab_rank:
        st.subheader("Rank Resumes Against Job Description")

        if not st.session_state.resumes:
            st.warning("⚠️ Please upload at least one resume in the 'Upload Resumes' tab.")
        elif not st.session_state.jd_text:
            st.warning("⚠️ Please provide a job description in the 'Job Description' tab.")
        else:
            if st.button("🚀 Rank Resumes", type="primary"):
                with st.spinner("Scoring and ranking candidates..."):
                    resume_texts = {
                        name: data["text"] for name, data in st.session_state.resumes.items()
                    }
                    results = rank_resumes(resume_texts, st.session_state.jd_text)
                    st.session_state.rank_results = results
                st.success("✅ Ranking complete!")

            if st.session_state.rank_results:
                st.markdown("### 🏆 Candidate Ranking")

                for rank, result in enumerate(st.session_state.rank_results, start=1):
                    parsed = st.session_state.resumes[result.candidate_name]["parsed"]

                    st.markdown(f"#### #{rank} — {parsed.name} ({result.candidate_name})")
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.progress(min(int(result.match_score), 100), text=f"Match Score: {result.match_score}%")

                    with col2:
                        experience_years = estimate_years_of_experience(parsed.experience)
                        num_certs = _count_certifications(parsed.certifications)
                        suitability = predict_suitability(
                            match_score=result.match_score,
                            years_experience=experience_years,
                            num_matching_skills=len(result.matching_skills),
                            num_certifications=num_certs,
                        )
                        badge = "🟢" if suitability.prediction == "Suitable" else "🔴"
                        st.metric("Suitability", f"{badge} {suitability.prediction}",
                                   f"{suitability.probability * 100:.1f}% confidence")

                    with col3:
                        classification = classify_resume(parsed.raw_text)
                        st.metric("Category", classification.category,
                                   f"{classification.confidence * 100:.1f}% confidence")

                    with st.expander("🔎 Skill Match Breakdown"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**✅ Matching Skills**")
                            st.write(", ".join(result.matching_skills) or "None")
                        with c2:
                            st.markdown("**❌ Missing Skills**")
                            st.write(", ".join(result.missing_skills) or "None")
                        with c3:
                            st.markdown("**➕ Extra Skills**")
                            st.write(", ".join(result.extra_skills) or "None")

                    st.markdown("---")

    # ================================================================
    # TAB 4: Candidate Details & AI Summary
    # ================================================================
    with tab_details:
        st.subheader("Candidate Details & AI-Generated Summary")

        if not st.session_state.resumes:
            st.info("Upload resumes first to view candidate details here.")
        else:
            candidate_filenames = list(st.session_state.resumes.keys())
            selected_file = st.selectbox(
                "Select a candidate",
                candidate_filenames,
                format_func=lambda f: f"{st.session_state.resumes[f]['parsed'].name} ({f})",
            )

            data = st.session_state.resumes[selected_file]
            parsed = data["parsed"]

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"### 👤 {parsed.name}")
                st.markdown(f"📧 {parsed.email}  |  📞 {parsed.phone}")
                st.markdown(f"**🛠️ Skills:** {', '.join(parsed.skills) or 'None detected'}")

            with col2:
                if st.button("✨ Generate AI Summary", key=f"summary_btn_{selected_file}"):
                    with st.spinner("Groq is writing a professional summary..."):
                        data["summary"] = summarize_resume(data["text"], parsed.name)

            if data["summary"]:
                st.markdown("#### ✍️ Professional Summary")
                st.markdown(f'<div class="hs-card">{data["summary"]}</div>', unsafe_allow_html=True)
