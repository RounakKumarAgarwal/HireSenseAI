# HireSenseAI

AI-powered recruitment platform for resume screening, ranking, and interview prep — built with Streamlit, scikit-learn, FAISS, and Groq LLM (RAG). Parses resumes, predicts candidate suitability, generates interview questions, and answers questions grounded in uploaded documents.

---

## 📌 Project Overview

Recruiters spend hours manually reading resumes, comparing them to job descriptions, and preparing interview questions. **HireSense AI** streamlines this entire workflow using a combination of classical Machine Learning (scikit-learn) and modern Generative AI (Groq LLM + Retrieval-Augmented Generation), so HR teams can:

- Screen and rank dozens of resumes against a job description in seconds
- Get an ML-backed "Suitable / Not Suitable" recommendation with a confidence score
- Automatically categorize candidates by role (Data Scientist, Web Developer, etc.)
- Generate professional candidate summaries and tailored interview questions instantly
- Chat with any resume, company policy document, or interview knowledge base — with answers grounded strictly in the uploaded documents (no hallucination)

This project was built as a college-level demonstration of a **production-style ML + RAG system**, with clean, modular, and well-commented code throughout.

---

## ✨ Features

| # | Feature | Description |
|---|---|---|
| 1 | **Resume Upload** | Upload one or multiple PDF resumes; text is extracted and displayed |
| 2 | **Resume Parsing** | Extracts Name, Email, Phone, Skills, Education, Experience, Certifications |
| 3 | **Skill Extraction** | Detects 50+ technical skills via regex matching; easy to extend |
| 4 | **Resume Summarization** | Groq LLM generates a 4-5 sentence professional summary |
| 5 | **Job Description Upload** | Paste text or upload a JD as PDF |
| 6 | **Resume Ranking** | TF-IDF + Cosine Similarity match score, matching/missing skills, ranked list |
| 7 | **Candidate Suitability Prediction** | Random Forest ML model predicts Suitable / Not Suitable + probability |
| 8 | **Resume Classification** | TF-IDF + Naive Bayes pipeline classifies resumes into 6 job categories |
| 9 | **AI Interview Question Generator** | Groq generates Technical, HR, Behavioral & Coding questions |
| 10 | **Resume Chatbot (RAG)** | Ask questions about a specific candidate; answers grounded in their resume only |
| 11 | **Company Policy Chatbot (RAG)** | Upload policy PDFs; ask HR policy questions grounded in those documents |
| 12 | **Interview Knowledge Base (RAG)** | RAG chatbot over a library of interview-prep guide PDFs |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Language | Python 3.12+ |
| Machine Learning | scikit-learn (Random Forest, Naive Bayes, TF-IDF) |
| NLP / Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| LLM | Groq API (Llama 3.1) |
| PDF Parsing | PyPDF2, pdfplumber |
| Data Handling | pandas, numpy |
| Visualization | matplotlib |
| Config | python-dotenv |

---

## 📁 Folder Structure

```
HireSense_AI/
├── app.py                          # Main Streamlit entry point + sidebar navigation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env.example                    # Environment variable template
│
├── assets/                         # Static assets (logos, images)
│
├── data/
│   ├── resumes/                    # (optional) persisted resume storage
│   ├── policies/                   # (optional) persisted policy PDFs
│   └── interview_questions/        # Interview knowledge base PDFs
│                                    # (e.g. "Java Interview Guide.pdf",
│                                    #  "Python Guide.pdf", "SQL Guide.pdf")
│
├── models/                         # Saved ML models (.joblib) - auto-generated
│
├── utils/                          # Core backend logic (no UI code)
│   ├── config.py                   # Paths, constants, env vars, logging
│   ├── pdf_parser.py                # PDF text extraction (PyPDF2 + pdfplumber)
│   ├── skill_extractor.py          # Technical skill detection
│   ├── resume_parser.py            # Structured resume field extraction
│   ├── matcher.py                   # TF-IDF resume-vs-JD matching
│   ├── ml_model.py                  # Random Forest suitability prediction
│   ├── classifier.py                # Resume category classification
│   ├── summarizer.py                # Groq client + resume summarization
│   ├── interview_generator.py      # Groq-based interview question generation
│   ├── rag_resume.py                # Shared RAG engine + resume chatbot
│   ├── rag_policy.py                # Company policy chatbot
│   └── rag_interview.py             # Interview knowledge base chatbot
│
└── pages/                          # One module per Streamlit screen
    ├── Home.py                     # HR dashboard landing page
    ├── Resume_Screening.py         # Upload, parse, rank, classify, summarize
    ├── Resume_Chat.py              # Per-candidate RAG chatbot
    ├── Policy_Chat.py              # Company policy RAG chatbot
    ├── Interview_Generator.py      # AI interview question generator
    ├── Analytics.py                # Charts and visual insights
    └── Model_Training.py           # Manual ML model retraining UI
```

> **Note on filenames:** the pages are implemented as `Resume_Screening.py`, `Resume_Chat.py`, etc. (underscores instead of spaces), because Python module names cannot contain spaces. The sidebar labels in the running app still read naturally, e.g. "📄 Resume Screening".

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
cd HireSense_AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Setup — Groq API Configuration

HireSense AI uses the **Groq API** for LLM features (summarization, interview question generation, and all RAG chatbot answers).

### Step 1 — Get a free Groq API key
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up / log in
3. Create a new API key

### Step 2 — Configure your environment
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste your key:
   ```
   GROQ_API_KEY=your_actual_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

> The app will still run without a Groq key — resume parsing, skill extraction, matching, ranking, ML suitability prediction, and classification all work fully offline. Only the LLM-powered features (summaries, interview questions, and chatbot answers) require a valid `GROQ_API_KEY`.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`) in your browser.

### First-run notes
- The two ML models (suitability predictor & resume classifier) train automatically on first use if no saved model is found in `models/`, and are cached to disk afterward.
- To enable the **Interview Knowledge Base** chatbot, place PDF guides (e.g. "Python Guide.pdf", "SQL Guide.pdf") into `data/interview_questions/` before launching, or use the upload option in the app.
- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically the first time a RAG chatbot is used — this requires an internet connection on first run only; it is cached locally afterward.

---

## 📊 Dataset

This project does **not** rely on any external hiring dataset. Instead:

- **Suitability Predictor**: trained on a synthetic dataset (`utils/ml_model.py → generate_sample_dataset()`) generated from a weighted, noisy formula combining match score, years of experience, matching skills, certifications, and education level — producing realistic, learnable hiring signal without needing real candidate data.
- **Resume Classifier**: trained on synthetic keyword phrases representative of each job category (`utils/classifier.py → CATEGORY_KEYWORDS`), combined and augmented to build a training corpus.

Both approaches are transparent, reproducible, and fully documented in code — ideal for a college project where explainability matters more than access to proprietary hiring data. You can inspect and retrain both models live from the **⚙️ Model Training** page in the app.

---

## 🖼️ Screenshots

> _Add screenshots of the running app here, e.g.:_
> - `assets/screenshot_home.png` — Home dashboard
> - `assets/screenshot_screening.png` — Resume Screening & Ranking
> - `assets/screenshot_chat.png` — Resume Chatbot
> - `assets/screenshot_analytics.png` — Analytics Dashboard

---

## 🚀 Future Scope

- Persist resumes, rankings, and chat history to a real database (PostgreSQL / MongoDB) instead of in-memory session state
- Replace regex-based resume parsing with a fine-tuned NER model (e.g. spaCy) for higher accuracy on non-standard resume formats
- Add authentication and multi-user HR team support
- Support scanned/image-based resumes via OCR (e.g. Tesseract)
- Add a candidate-facing portal for status tracking
- Expand the suitability model with real historical hiring outcomes
- Add support for additional LLM providers (OpenAI, Anthropic) as alternatives to Groq
- Export ranked candidate reports and interview question sets as PDF/Word documents

---

## 📄 License

This project was built for educational purposes as a college project.
