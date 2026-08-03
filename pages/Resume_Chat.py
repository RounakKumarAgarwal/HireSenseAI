"""
Resume_Chat.py
===============
The Resume Chatbot (RAG) page. HR selects a candidate and asks
natural-language questions about their resume - e.g. "What projects
has this candidate completed?", "Does this candidate know SQL?",
"How many years of experience?". Answers are generated using
Retrieval-Augmented Generation and are grounded ONLY in that
candidate's resume text.
"""

from __future__ import annotations

import streamlit as st

from utils.rag_resume import ResumeRAGEngine

SUGGESTED_QUESTIONS = [
    "What projects has this candidate completed?",
    "Does this candidate know SQL?",
    "How many years of experience does this candidate have?",
    "What certifications does this candidate have?",
]


def _get_or_build_engine(filename: str) -> ResumeRAGEngine:
    """Lazily build (and cache) the RAG engine for a given candidate."""
    data = st.session_state.resumes[filename]

    if data["rag_engine"] is None:
        with st.spinner("Indexing resume for chat (one-time per candidate)..."):
            engine = ResumeRAGEngine(candidate_name=data["parsed"].name)
            engine.index_resume(data["text"])
            data["rag_engine"] = engine

    return data["rag_engine"]


def render() -> None:
    """Render the Resume Chat page."""

    st.title("💬 Resume Chatbot (RAG)")
    st.caption("Ask questions about a specific candidate. Answers are grounded only in their resume.")
    st.markdown("---")

    if not st.session_state.resumes:
        st.info("📄 Upload resumes on the **Resume Screening** page first to chat with them here.")
        return

    candidate_filenames = list(st.session_state.resumes.keys())
    selected_file = st.selectbox(
        "Select a candidate to chat about",
        candidate_filenames,
        format_func=lambda f: f"{st.session_state.resumes[f]['parsed'].name} ({f})",
    )

    # Make sure this candidate has a chat history slot
    resume_chats = st.session_state.chat_history["resume_chat"]
    if selected_file not in resume_chats:
        resume_chats[selected_file] = []

    engine = _get_or_build_engine(selected_file)

    st.markdown("**💡 Try asking:**")
    suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_suggestion = None
    for col, question in zip(suggestion_cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(question, key=f"suggest_{selected_file}_{question}", use_container_width=True):
                clicked_suggestion = question

    st.markdown("---")

    # ----------------------------------------------------------------
    # Render existing chat history
    # ----------------------------------------------------------------
    for role, message in resume_chats[selected_file]:
        with st.chat_message(role):
            st.markdown(message)

    # ----------------------------------------------------------------
    # Handle new input (either typed or a clicked suggestion)
    # ----------------------------------------------------------------
    user_question = st.chat_input("Ask a question about this candidate's resume...")
    final_question = clicked_suggestion or user_question

    if final_question:
        resume_chats[selected_file].append(("user", final_question))
        with st.chat_message("user"):
            st.markdown(final_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the resume..."):
                answer = engine.ask(final_question)
            st.markdown(answer)

        resume_chats[selected_file].append(("assistant", answer))

    if resume_chats[selected_file]:
        if st.button("🗑️ Clear Chat History"):
            resume_chats[selected_file] = []
            st.rerun()
