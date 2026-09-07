"""
Job Analysis Agent
-------------------
Responsibility: take a raw job description (plain text) and extract a
structured JSON representation of its requirements:

    {
        "role_title": str,
        "seniority": str,              # e.g. "Entry-level", "Mid", "Senior"
        "required_skills": [str],
        "preferred_skills": [str],
        "technologies": [str],
        "responsibilities": [str],
        "experience_required": str,    # e.g. "2-4 years"
        "education_required": str,
        "keywords": [str],             # general ATS-relevant terms
        "industry": str
    }

This output is consumed by:
- Match Score Agent (Phase 3): compares against the Career Vault
- Resume Tailoring Agent (Phase 4): decides what to emphasize/reorder
- ATS Optimizer Agent (Phase 5): checks keyword coverage
"""

import json
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

EXTRACTION_PROMPT = """You are a precise job description parsing engine.

Given the job description text below, extract a structured JSON object with
these exact keys:
role_title, seniority, required_skills, preferred_skills, technologies,
responsibilities, experience_required, education_required, keywords, industry

Rules:
- "required_skills", "preferred_skills", "technologies", "responsibilities",
  and "keywords" MUST be lists of strings.
- "role_title", "seniority", "experience_required", "education_required",
  and "industry" MUST be strings (empty string if not mentioned).
- "seniority" must be one of: "Internship", "Entry-level", "Mid-level",
  "Senior", "Lead", "Unspecified" -- pick the best fit.
- "keywords" should include general ATS-relevant terms beyond the explicit
  skill list (e.g. methodologies, certifications, soft skills mentioned).
- Do NOT invent requirements not present in the text.
- Return ONLY valid JSON, no markdown fences, no commentary.

Job description text:
---
{jd_text}
---
"""


def analyze_job_description(jd_text: str) -> dict:
    """Use an LLM to convert raw JD text into structured requirements JSON."""
    if not jd_text.strip():
        raise ValueError("Job description text is empty.")

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    prompt = EXTRACTION_PROMPT.format(jd_text=jd_text)
    response = llm.invoke(prompt)

    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw output: {content}")


def analyze_job_description_from_file(file_path: str) -> dict:
    """Convenience wrapper: read a .txt JD file and analyze it."""
    with open(file_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
    return analyze_job_description(jd_text)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_jd.txt"
    result = analyze_job_description_from_file(path)
    print(json.dumps(result, indent=2))