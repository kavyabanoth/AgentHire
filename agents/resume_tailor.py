"""
Resume Tailoring Agent
------------------------
Responsibility: given the original structured resume, the JD requirements
(Phase 2), and the match score breakdown (Phase 3), produce a TAILORED
version of the resume:

    {
        "summary": str,                 # rewritten to target the role
        "skills": [str],                # reordered: relevant skills first
        "experience": [...],            # same entries, bullets re-emphasized
        "projects": [...],              # same entries, bullets re-emphasized
        "education": [...],             # unchanged
        "certifications": [...],        # unchanged
        "tailoring_notes": [str]        # what was changed and why
    }

Grounding strategy: before generating anything, this agent retrieves the
resume's own Career Vault chunks (Phase 1) most relevant to the JD's
requirements, and feeds ONLY that retrieved + original content to the LLM.
This is the "career vault" pattern -- it prevents hallucinated experience
because the model can only work with real, retrieved material.

Hard rule enforced in the prompt: never claim a skill listed in
match_result["missing_skills"] as something the candidate has.
"""

import json
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL
from career_vault.ingest import query_career_vault

TAILOR_PROMPT = """You are an expert, ethical resume writer. You tailor resumes to
specific job descriptions WITHOUT ever inventing experience, skills, or
qualifications the candidate does not have.

CANDIDATE'S ORIGINAL RESUME (ground truth -- do not add anything beyond this):
---
{original_resume}
---

RETRIEVED RELEVANT CONTEXT FROM CANDIDATE'S CAREER HISTORY (further ground truth):
---
{retrieved_context}
---

TARGET JOB DESCRIPTION REQUIREMENTS:
---
{jd_requirements}
---

SKILLS THE CANDIDATE IS MISSING (per match analysis -- NEVER claim these as possessed,
never add them to skills or imply experience with them):
---
{missing_skills}
---

Your task: produce a tailored version of this resume as JSON with these exact keys:
summary, skills, experience, education, projects, certifications, tailoring_notes

Rules:
1. NEVER invent, exaggerate, or imply any skill, tool, or experience not present
   in the original resume or retrieved context above.
2. NEVER include anything from the "missing skills" list as something possessed.
3. Reorder "skills" so the ones matching the job requirements appear first;
   keep all original skills, just reordered.
4. Rewrite "summary" (2-3 sentences) to highlight genuinely relevant experience
   for this specific role -- based only on what's in the original resume.
5. For "experience" and "projects": keep the same entries (same title/company/
   dates), but you may lightly rephrase bullets to emphasize relevance and use
   stronger action verbs. Do not change facts, numbers, or add new bullets with
   unverifiable claims.
6. "education" and "certifications": copy unchanged from the original resume.
7. "tailoring_notes": a short list of strings explaining what you changed and why
   (e.g. "Reordered skills to lead with SQL and Python per JD priority").
8. Return ONLY valid JSON, no markdown fences, no commentary.
"""


def _gather_retrieved_context(jd: dict, user_id: str, n_results: int = 3) -> str:
    """Query the Career Vault for content relevant to the JD's top requirements,
    to ground the tailoring LLM call in real retrieved material."""
    queries = jd.get("required_skills", [])[:5] + jd.get("responsibilities", [])[:3]
    seen = set()
    context_chunks = []

    for query in queries:
        try:
            hits = query_career_vault(query, user_id=user_id, n_results=n_results)
        except Exception:
            continue  # career vault may not exist yet; fall back to original resume only
        docs = hits.get("documents", [[]])[0]
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                context_chunks.append(doc)

    return "\n---\n".join(context_chunks) if context_chunks else "(no additional context retrieved)"


def tailor_resume(resume: dict, jd: dict, match_result: dict, user_id: str = "default_user") -> dict:
    retrieved_context = _gather_retrieved_context(jd, user_id)
    missing_skills = [s["skill"] for s in match_result.get("missing_skills", [])]

    jd_requirements_summary = {
        "role_title": jd.get("role_title", ""),
        "required_skills": jd.get("required_skills", []),
        "preferred_skills": jd.get("preferred_skills", []),
        "responsibilities": jd.get("responsibilities", []),
        "keywords": jd.get("keywords", []),
    }

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    prompt = TAILOR_PROMPT.format(
        original_resume=json.dumps(resume, indent=2),
        retrieved_context=retrieved_context,
        jd_requirements=json.dumps(jd_requirements_summary, indent=2),
        missing_skills=json.dumps(missing_skills),
    )
    response = llm.invoke(prompt)

    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1)

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

    resume = parse_resume(resume_path)
    jd = analyze_job_description_from_file(jd_path)
    match_result = calculate_match_score(resume, jd)
    tailored = tailor_resume(resume, jd, match_result, user_id=user_id)
    print(json.dumps(tailored, indent=2))