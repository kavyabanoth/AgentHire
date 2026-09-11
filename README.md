# 🎯 AgentHire AI

**Multi-agent resume tailoring & job-match platform** — powered by LangGraph, Groq (Llama), and ChromaDB.

<img width="1890" height="891" alt="Screenshot 2026-09-11 094349" src="https://github.com/user-attachments/assets/b55414da-b1c6-4624-a661-4a9a0c592b87" />
<img width="1907" height="892" alt="Screenshot 2026-09-11 094445" src="https://github.com/user-attachments/assets/c597ad8e-488a-45f1-80b2-0ebe28c67425" />
<img width="1395" height="828" alt="Screenshot 2026-09-11 094523" src="https://github.com/user-attachments/assets/4b58925b-7765-41ca-968d-25ac748bf1d5" />
<img width="911" height="666" alt="Screenshot 2026-09-11 094555" src="https://github.com/user-attachments/assets/fb09bab8-8857-4dc2-98c0-15e5465c8e38" />
<img width="1322" height="772" alt="Screenshot 2026-09-11 094705" src="https://github.com/user-attachments/assets/0f3d48b8-b885-40f3-b5ad-da5e8d561c1a" />

## 🚀 Live Demo

**Try it here:** https://agenthire-appwu5ixwm62sfdmltykemd.streamlit.app

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agenthire-appwu5ixwm62sfdmltykemd.streamlit.app)

Upload any resume (PDF/DOCX/TXT), paste any job description, and get:
- A quantified match score with a skills/experience/education breakdown
- A tailored, grounded resume (no fabricated experience)
- An ATS keyword-coverage report
- A personalized cover letter

---

## 📖 Overview

AgentHire AI is an end-to-end multi-agent system that automates job-application preparation while staying **grounded and truthful** — every generated resume bullet, match score, and cover letter claim is traceable back to the candidate's real, uploaded resume. No fabricated skills, no invented experience.

The system is built around a **"career vault"** pattern: the candidate's resume is parsed, chunked, and embedded into a vector store, and every downstream agent (tailoring, cover letters) retrieves from that vault before generating anything — rather than letting an LLM freely hallucinate content.

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │  Streamlit UI   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ LangGraph Graph │  (orchestrator.py)
                    └────────┬────────┘
                             │
   ┌──────────┬──────────┬──┴───────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼          ▼
Resume    Career    Job        Match     Resume     ATS
Parser →  Vault  →  Analyzer → Score  →  Tailor  →  Optimizer → Cover Letter
Agent     (ChromaDB) Agent     Agent     Agent      Agent        Agent
```

Each agent is a LangGraph node; a shared `AgentState` flows through the graph, accumulating results at each step (see `orchestrator.py`).

## 🧩 The 6 Agents

| # | Agent | File | Responsibility |
|---|-------|------|-----------------|
| 1 | **Resume Parser** | `agents/resume_parser.py` | PDF/DOCX/TXT → structured JSON (summary, experience, education, skills, projects) |
| 2 | **Career Vault** | `career_vault/ingest.py` | Chunks + embeds resume into ChromaDB for grounded retrieval |
| 3 | **Job Analyzer** | `agents/job_analyzer.py` | Job description → structured requirements (skills, seniority, keywords) |
| 4 | **Match Score** | `agents/match_score.py` | Hybrid keyword + embedding scoring → overall %, matched/missing skills |
| 5 | **Resume Tailor** | `agents/resume_tailor.py` | Retrieves grounded context, rewrites/reorders resume for the target JD |
| 6 | **ATS Optimizer** | `agents/ats_optimizer.py` | Literal, rule-based ATS scoring (mirrors how real ATS software parses resumes) |
| 7 | **Cover Letter** | `agents/cover_letter.py` | Generates a grounded cover letter referencing real achievements |

## 🛠️ Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Groq (Llama-based `openai/gpt-oss-120b`) — free tier
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`) — local, no API cost
- **Vector DB:** ChromaDB (persistent local store)
- **Resume parsing:** pdfplumber, python-docx
- **UI:** Streamlit
- **Language:** Python 3.11

## 📐 Design Decisions Worth Noting

**Two different scoring philosophies, on purpose:**
- The **Match Score Agent** (Phase 3) uses semantic embedding similarity — it understands that "dashboard experience" relates to "data visualization" even without exact word overlap.
- The **ATS Optimizer** (Phase 5) deliberately uses **literal keyword matching** instead — because that's how real-world ATS software (Workday, Greenhouse, etc.) actually parses resumes. It doesn't "understand" synonyms.

This gap is intentional and demonstrates a real, well-documented problem in ATS systems: a resume can be semantically strong yet still fail a literal keyword scan. The project surfaces this gap explicitly rather than hiding it.

**Grounding over generation:** every agent that produces resume/cover-letter content retrieves from the Career Vault first and is explicitly instructed never to claim a skill flagged as "missing" by the Match Score Agent.

**Scope boundaries:** job search / scraping and browser automation (auto-apply) were intentionally excluded — these carry ToS and legal risk and are lower-value than tailoring quality. This mirrors the "Not Included" section of the original project spec.

## 📁 Project Structure

```
agenthire_ai/
├── agents/
│   ├── resume_parser.py
│   ├── job_analyzer.py
│   ├── match_score.py
│   ├── resume_tailor.py
│   ├── ats_optimizer.py
│   └── cover_letter.py
├── career_vault/
│   └── ingest.py
├── data/                      # sample resumes/JDs, ChromaDB store (gitignored)
├── config.py                  # model + API key config (env or Streamlit secrets)
├── orchestrator.py            # LangGraph pipeline definition
├── streamlit_app.py           # UI
├── test_phase1.py … test_phase7.py   # incremental phase-by-phase test scripts
├── requirements.txt
├── runtime.txt                # pins Python 3.11 for deployment
└── README.md
```

## ⚙️ Setup (Local)

```bash
git clone https://github.com/kavyabanoth/AgentHire.git
cd AgentHire/agenthire_ai
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
cp .env.example .env               # then add your GROQ_API_KEY
streamlit run streamlit_app.py
```

Get a free Groq API key at https://console.groq.com/keys (no payment method required).

## 🧪 Testing Incrementally

Each phase has a standalone test script, useful for demoing the system's build-up:

```bash
python test_phase1.py "data/resume.pdf"                          # parse + ingest
python test_phase2.py data/sample_jd.txt                         # JD analysis
python test_phase3.py "data/resume.pdf" data/sample_jd.txt        # match scoring
python test_phase4.py "data/resume.pdf" data/sample_jd.txt kavya  # tailoring
python test_phase5.py "data/resume.pdf" data/sample_jd.txt kavya  # ATS check
python test_phase6.py "data/resume.pdf" data/sample_jd.txt kavya "Company"  # cover letter
python test_phase7.py "data/resume.pdf" data/sample_jd.txt kavya "Company"  # full LangGraph pipeline
```

- ## ⚠️ Known Limitations

- **The app sleeps after inactivity** — Streamlit Community Cloud's free tier puts apps to sleep after a period without visitors. The first person to open the link after a while will see a "This app has gone to sleep" screen and need to click a button to wake it; the app then takes 30-60 seconds to restart. This is a hosting-tier limitation, not an application bug — subsequent visits are fast until it sleeps again.
- **Career Vault doesn't persist across Streamlit Cloud restarts** — ChromaDB writes to local disk, which is ephemeral on the free hosting tier. Each pipeline run re-ingests fresh, so functionality is unaffected, but there's no long-term memory across sessions in the deployed version.
- **Education scoring** (within Match Score) uses a single embedding comparison — the least sophisticated of the three sub-scores.
- **Free-tier LLM output** occasionally has minor formatting glitches (e.g. dropped spaces at line-wrap boundaries).
- **No job search, browser automation, or learning-from-outcomes agents** — intentionally out of scope (see Design Decisions above).

## 🔭 Future Scope

- Job Search Agent (LinkedIn/Greenhouse/Lever integration)
- Learning Agent (adapts strategy based on application outcomes)
- Browser Automation Agent (with mandatory human approval before submission)
- Multi-resume comparison / recruiter-facing ranking mode
- Persistent, multi-user Career Vault (migrate to a hosted vector DB)

## 📄 License

Built as a final-year academic project.
