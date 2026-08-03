"""
utils package
==============
Core backend modules for HireSense AI:
    - config.py              : paths, constants, env vars, logging
    - pdf_parser.py           : PDF text extraction
    - skill_extractor.py      : technical skill detection
    - resume_parser.py        : structured resume field extraction
    - matcher.py               : TF-IDF resume vs JD matching
    - ml_model.py              : Random Forest suitability prediction
    - classifier.py            : resume category classification
    - summarizer.py            : Groq-based resume summarization
    - interview_generator.py  : Groq-based interview question generation
    - rag_resume.py            : FAISS + SentenceTransformers resume chatbot
    - rag_policy.py            : FAISS-based company policy chatbot
    - rag_interview.py         : FAISS-based interview knowledge base chatbot
"""
