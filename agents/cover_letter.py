"""
Cover Letter Agent
--------------------
Responsibility: given the resume, JD, and match score breakdown, generate a
personalized cover letter that:
- references SPECIFIC real achievements/projects from the resume (grounded
  via Career Vault retrieval, same pattern as the Resume Tailoring Agent)
- explains genuine fit using the matched_skills from Phase 3
- addresses gaps (missing_skills) honestly via transferable skills or
  learning intent -- NEVER by claiming experience that isn't there
- stays concise (3-4 paragraphs) and avoids generic filler phrases

Output: {"cover_letter": str, "key_points_used": [str]}
"""

import json
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL
from career_vault.ingest import query_career_vault

COVER_LETTER_PROMPT = """You are an expert cover letter writer. You write concise,
specific, honest cover letters -- never generic filler, never fabricated experience.

CANDIDATE'S RESUME (ground truth):
---
{resume}
---

RETRIEVED RELEVANT CONTEXT FROM CANDIDATE'S CAREER HISTORY:
---
{retrieved_context}
---

TARGET ROLE: {role_title} at {company_name}

JOB RESPONSIBILITIES:
---
{responsibilities}
---

SKILLS THE CANDIDATE GENUINELY MATCHES (lean into these):
---
{matched_skills}
---

SKILLS THE CANDIDATE IS MISSING (do NOT claim these -- if addressed at all, frame
honestly as eagerness to learn or a related transferable skill, never as experience):
---
{missing_skills}
---

Write a cover letter as JSON with these exact keys: cover_letter, key_points_used

Rules:
1. "cover_letter" is a 3-4 paragraph professional letter (no letterhead/date needed,
   start directly with "Dear Hiring Manager," or similar).
2. Reference at least 2 SPECIFIC real projects/experiences from the resume/context
   above -- not vague claims like "I have strong analytical skills."
3. NEVER claim experience with a missing skill. You may express genuine interest in
   learning it, but do not imply you already have it.
4. NO generic filler phrases like "I am a hardworking team player" without backing
   it with a specific example.
5. End with a brief, confident closing paragraph (not desperate or overly formal).
6. "key_points_used": a list of the specific real achievements/projects referenced,
   so the candidate can verify grounding.
7. Return ONLY valid JSON, no markdown fences, no commentary.
"""


def _gather_retrieved_context(jd: dict, user_id: str, n_results: int = 3) -> str:
    queries = jd.get("required_skills", [])[:4] + jd.get("responsibilities", [])[:3]
    seen = set()
    context_chunks = []

    for query in queries:
        try:
            hits = query_career_vault(query, user_id=user_id, n_results=n_results)
        except Exception:
            continue
        docs = hits.get("documents", [[]])[0]
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                context_chunks.append(doc)

    return "\n---\n".join(context_chunks) if context_chunks else "(no additional context retrieved)"


def generate_cover_letter(
    resume: dict,
    jd: dict,
    match_result: dict,
    company_name: str = "your company",
    user_id: str = "default_user",
) -> dict:
    retrieved_context = _gather_retrieved_context(jd, user_id)

    matched_skills = [s["skill"] for s in match_result.get("matched_skills", [])]
    missing_skills = [s["skill"] for s in match_result.get("missing_skills", [])]

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.4)
    prompt = COVER_LETTER_PROMPT.format(
        resume=json.dumps(resume, indent=2),
        retrieved_context=retrieved_context,
        role_title=jd.get("role_title", "the role"),
        company_name=company_name,
        responsibilities=json.dumps(jd.get("responsibilities", [])),
        matched_skills=json.dumps(matched_skills),
        missing_skills=json.dumps(missing_skills),
    )
    response = llm.invoke(prompt)

    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw output: {content}")


if __name__ == "__main__":
    import sys
    from agents.resume_parser import parse_resume
    from agents.job_analyzer import analyze_job_description_from_file
    from agents.match_score import calculate_match_score

    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "default_user"
    company_name = sys.argv[4] if len(sys.argv) > 4 else "your company"

    resume = parse_resume(resume_path)
    jd = analyze_job_description_from_file(jd_path)
    match_result = calculate_match_score(resume, jd)
    result = generate_cover_letter(resume, jd, match_result, company_name=company_name, user_id=user_id)
    print(json.dumps(result, indent=2))