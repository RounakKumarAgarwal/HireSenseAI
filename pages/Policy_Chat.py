"""
Policy_Chat.py
===============
The Company Policy Chatbot page. HR uploads one or more policy PDFs
(leave policy, code of conduct, WFH policy, etc.) and can then ask
natural-language questions. Answers are generated using RAG and are
grounded ONLY in the uploaded policy documents.
"""

from __future__ import annotations

import streamlit as st

from utils.pdf_parser import extract_text_from_pdf
from utils.rag_policy import PolicyRAGEngine

SUGGESTED_QUESTIONS = [
    "How many paid leaves are employees entitled to?",
    "What is the work-from-home policy?",
    "What is the notice period for resignation?",
]


def render() -> None:
    """Render the Policy Chat page."""

    st.title("📜 Company Policy Chatbot")
    st.caption("Upload HR policy documents and ask questions - answers are grounded only in what you upload.")
    st.markdown("---")

    # ----------------------------------------------------------------
    # Upload section
    # ----------------------------------------------------------------
    st.subheader("📤 Upload Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more policy PDFs (e.g. Leave Policy, Code of Conduct)",
        type=["pdf"],
        accept_multiple_files=True,
        key="policy_uploader",
    )

    if uploaded_files:
        new_files_added = False
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.policy_documents:
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    text = extract_text_from_pdf(uploaded_file)
                st.session_state.policy_documents[uploaded_file.name] = text
                new_files_added = True

        if new_files_added:
            with st.spinner("Building policy knowledge index..."):
                engine = PolicyRAGEngine()
                engine.index_policies(st.session_state.policy_documents)
                st.session_state.policy_engine = engine
            st.success(f"✅ Indexed {len(st.session_state.policy_documents)} policy document(s).")

    if st.session_state.policy_documents:
        with st.expander(f"📁 {len(st.session_state.policy_documents)} document(s) indexed"):
            for filename in st.session_state.policy_documents:
                st.markdown(f"- {filename}")

        if st.button("🗑️ Clear All Policy Documents"):
            st.session_state.policy_documents = {}
            st.session_state.policy_engine = None
            st.session_state.chat_history["policy_chat"] = []
            st.rerun()

    st.markdown("---")

    # ----------------------------------------------------------------
    # Chat section
    # ----------------------------------------------------------------
    st.subheader("💬 Ask a Policy Question")

    if st.session_state.policy_engine is None:
        st.info("Upload at least one policy PDF above to enable the chatbot.")
        return

    st.markdown("**💡 Try asking:**")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_suggestion = None
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(question, key=f"policy_suggest_{question}", use_container_width=True):
                clicked_suggestion = question

    st.markdown("---")

    history = st.session_state.chat_history["policy_chat"]
    for role, message in history:
        with st.chat_message(role):
            st.markdown(message)

    user_question = st.chat_input("Ask a question about your company policies...")
    final_question = clicked_suggestion or user_question

    if final_question:
        history.append(("user", final_question))
        with st.chat_message("user"):
            st.markdown(final_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                answer = st.session_state.policy_engine.ask(final_question)
            st.markdown(answer)

        history.append(("assistant", answer))

    if history:
        if st.button("🗑️ Clear Chat History", key="policy_clear_chat"):
            st.session_state.chat_history["policy_chat"] = []
            st.rerun()
