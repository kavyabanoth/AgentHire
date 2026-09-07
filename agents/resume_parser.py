"""
Resume Parser Agent
--------------------
Responsibility: take a raw resume file (PDF / DOCX / TXT) and produce
a structured JSON representation:
    {
        "summary": str,
        "experience": [{"title": str, "company": str, "duration": str, "bullets": [str]}],
        "education": [{"degree": str, "institution": str, "duration": str}],
        "skills": [str],
        "projects": [{"title": str, "description": str, "tech": [str]}],
        "certifications": [str]
    }

This structured object becomes the "master resume" that later gets chunked
and embedded into the Career Vault (see career_vault/ingest.py).
"""

import json
import pdfplumber
import docx
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

EXTRACTION_PROMPT = """You are a precise resume parsing engine.
Given the raw resume text below, extract a structured JSON object with these keys:
summary, experience, education, skills, projects, certifications.

Rules:
- Do NOT invent or infer information not present in the text.
- If a field is missing, return an empty list or empty string for it.
- "experience" MUST be a list of objects, each with keys: title, company, duration, bullets (list of strings). Do NOT return experience as plain strings.
- "projects" MUST be a list of objects, each with keys: title, description, tech (list of strings). Do NOT return projects as plain strings.
- "education" MUST be a list of objects, each with keys: degree, institution, duration.
- Bullets must be copied/lightly cleaned, not rewritten.
- Return ONLY valid JSON, no markdown fences, no commentary.

Resume text:
---
{resume_text}
---
"""


def extract_raw_text(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT resume files."""
    if file_path.lower().endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def structure_resume(raw_text: str) -> dict:
    """Use an LLM to convert raw resume text into structured JSON."""
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    prompt = EXTRACTION_PROMPT.format(resume_text=raw_text)
    response = llm.invoke(prompt)

    content = response.content.strip()
    # Defensive cleanup in case the model wraps output in ```json fences
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw output: {content}")


def parse_resume(file_path: str) -> dict:
    """End-to-end: file path -> structured resume dict."""
    raw_text = extract_raw_text(file_path)
    if not raw_text.strip():
        raise ValueError("No extractable text found in resume file.")
    return structure_resume(raw_text)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    result = parse_resume(path)
    print(json.dumps(result, indent=2))