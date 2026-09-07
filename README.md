# AgentHire AI — Phase 1 Scaffold
<img width="1920" height="908" alt="Screenshot 2026-09-07 171030" src="https://github.com/user-attachments/assets/0a7f1441-fb75-49f7-aafb-8ee0cf08967a" />

Multi-agent resume tailoring & job-match platform. This scaffold covers
**Phase 1**: Resume Parser Agent + Career Vault ingestion (ChromaDB).

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Structure
```
agenthire_ai/
├── agents/
│   └── resume_parser.py     # PDF/DOCX/TXT -> structured JSON resume
├── career_vault/
│   └── ingest.py             # structured resume -> ChromaDB (chunked + embedded)
├── data/                     # put sample resumes here
├── config.py                 # model + DB config
├── test_phase1.py            # smoke test: parse -> ingest -> retrieve
└── requirements.txt
```

## Run
```bash
python test_phase1.py data/sample_resume.pdf
```

This will:
1. Parse the resume into structured sections (summary, experience, education, skills, projects, certifications)
2. Chunk and embed each section into a ChromaDB collection ("career_vault")
3. Run a sample retrieval query to confirm grounding works

## Next Phases
- **Phase 2**: Job Analyzer Agent (JD -> skills/keywords JSON)
- **Phase 3**: Match Score Agent (embedding similarity + keyword gap)
- **Phase 4**: Resume Tailoring Agent (retrieves from Career Vault, rewrites bullets, grounded — no hallucination)
- **Phase 5**: ATS Optimizer Agent
- **Phase 6**: Cover Letter Agent
- **Phase 7**: LangGraph orchestration wiring all agents
- **Phase 8**: Streamlit dashboard + deployment
