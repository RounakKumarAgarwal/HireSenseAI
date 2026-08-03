"""
app.py
======
Main entry point for the HireSense AI Streamlit application.

This file is responsible for:
    1. Configuring the Streamlit page (title, icon, layout)
    2. Injecting shared custom CSS for a polished, professional look
    3. Initializing all shared session-state objects (so every page
       can read/write the same in-memory data without re-computing it)
    4. Rendering the sidebar navigation
    5. Routing to the correct page module based on the sidebar selection

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from pages import (
    Home,
    Resume_Screening,
    Resume_Chat,
    Policy_Chat,
    Interview_Generator,
    Analytics,
    Model_Training,
)

# ------------------------------------------------------------------
# 1. Page configuration (must be the first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="HireSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 2. Shared custom CSS
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Overall app background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Headings */
    h1, h2, h3 {
        color: #111827;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Custom card container */
    .hs-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. Initialize shared session state
# ------------------------------------------------------------------
def init_session_state() -> None:
    """
    Create all shared session-state keys the first time the app
    loads, so every page module can safely assume they exist.
    """
    defaults = {
        # {filename: {"text": str, "parsed": ParsedResume, "rag_engine": ResumeRAGEngine}}
        "resumes": {},
        # Currently pasted / uploaded job description text
        "jd_text": "",
        # Cached list[MatchResult] from the last ranking run
        "rank_results": [],
        # Candidate currently selected in the Resume Chat page
        "selected_candidate": None,
        # PolicyRAGEngine instance (built once policies are uploaded)
        "policy_engine": None,
        # {filename: extracted_text} for uploaded policy PDFs
        "policy_documents": {},
        # InterviewKnowledgeBaseRAGEngine instance
        "interview_kb_engine": None,
        # Chat history per page. resume_chat is keyed per-candidate since
        # each candidate has their own isolated RAG engine; policy_chat and
        # interview_kb_chat are single global conversations.
        "chat_history": {"resume_chat": {}, "policy_chat": [], "interview_kb_chat": []},
        # Cached generated interview questions (InterviewQuestions dataclass)
        "interview_questions": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ------------------------------------------------------------------
# 4. Sidebar navigation
# ------------------------------------------------------------------
PAGES = {
    "🏠 Home": Home,
    "📄 Resume Screening": Resume_Screening,
    "💬 Resume Chat": Resume_Chat,
    "📜 Policy Chat": Policy_Chat,
    "🎤 Interview Generator": Interview_Generator,
    "📊 Analytics": Analytics,
    "⚙️ Model Training": Model_Training,
}

with st.sidebar:
    st.markdown("## 🧠 HireSense AI")
    st.caption("Intelligent Recruitment & Candidate Evaluation")
    st.markdown("---")

    selection = st.radio(
        label="Navigate",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(f"📁 Resumes loaded: **{len(st.session_state.resumes)}**")
    st.caption("Built with Streamlit, scikit-learn, FAISS & Groq")

# ------------------------------------------------------------------
# 5. Route to the selected page
# ------------------------------------------------------------------
PAGES[selection].render()
