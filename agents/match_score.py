"""
Match Score Agent
-------------------
Responsibility: given a structured resume (Phase 1 output) and a structured
job description (Phase 2 output), compute:

    {
        "overall_match": float,        # 0-100
        "skills_score": float,         # 0-100
        "experience_score": float,     # 0-100
        "education_score": float,      # 0-100
        "matched_skills": [{"skill": str, "score": float}],
        "missing_skills": [{"skill": str, "score": float}],
        "verdict": str                 # "High" / "Medium" / "Low" chance
    }

Uses the same local sentence-transformers embedding model as the Career
Vault (Phase 1), so no extra API calls or cost -- pure vector math.
"""

import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from config import EMBEDDING_MODEL

# Thresholds -- tunable
SKILL_MATCH_THRESHOLD = 0.40   # combined hybrid score above this = "matched"
WEIGHT_SKILLS = 0.55
WEIGHT_EXPERIENCE = 0.30
WEIGHT_EDUCATION = 0.15

_model = None  # lazy-loaded singleton so repeated calls don't reload the model


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _display_name(skill: str) -> str:
    """Strip parenthetical examples for display only, e.g. 'Python (pandas, numpy)' -> 'Python'."""
    return re.sub(r"\s*\(.*?\)", "", skill).strip()


def _keyword_score(skill_text: str, full_text_lower: str) -> float:
    """Fraction of the skill's meaningful words found (as whole words) in the resume text.
    Direct substring match of the full phrase scores 1.0."""
    skill_lower = skill_text.lower()
    if skill_lower in full_text_lower:
        return 1.0
    words = re.findall(r"[a-z0-9\+\#\.]+", skill_lower)
    words = [w for w in words if w not in {"and", "or", "with", "the", "a", "of", "in", "for"}]
    if not words:
        return 0.0
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", full_text_lower))
    return hits / len(words)


def _resume_corpus(resume: dict) -> list[str]:
    """Flatten resume into a list of text chunks for embedding comparison."""
    chunks = []

    if resume.get("summary"):
        chunks.append(resume["summary"])

    for exp in resume.get("experience", []):
        if isinstance(exp, dict):
            bullets = " ".join(exp.get("bullets", []))
            chunks.append(f"{exp.get('title', '')} {exp.get('company', '')} {bullets}")
        else:
            chunks.append(str(exp))

    for proj in resume.get("projects", []):
        if isinstance(proj, dict):
            tech = ", ".join(proj.get("tech", []))
            chunks.append(f"{proj.get('title', '')} {proj.get('description', '')} {tech}")
        else:
            chunks.append(str(proj))

    skills = resume.get("skills", [])
    if skills:
        chunks.append(", ".join(skills) if isinstance(skills, list) else str(skills))

    for edu in resume.get("education", []):
        chunks.append(edu if isinstance(edu, str) else json_safe_str(edu))

    return [c for c in chunks if c.strip()]


def json_safe_str(obj) -> str:
    """Flatten a dict education entry into a plain string for embedding."""
    if isinstance(obj, dict):
        return " ".join(str(v) for v in obj.values())
    return str(obj)


def _skill_match_scores(skills: list[str], resume_chunk_embeddings, resume_chunks: list[str], full_text_lower: str, model) -> list[dict]:
    """For each JD skill, combine keyword overlap with best embedding similarity
    against any resume chunk. Uses the FULL skill text (including parenthetical
    alternatives like 'Power BI or Tableau') for matching, since those specific
    terms are often the most matchable part of a skill entry."""
    results = []
    for skill in skills:
        keyword_score = _keyword_score(skill, full_text_lower)

        skill_embedding = model.encode(skill, convert_to_tensor=True)
        sims = cos_sim(skill_embedding, resume_chunk_embeddings)[0]
        embedding_score = float(sims.max()) if len(sims) > 0 else 0.0

        # Keyword match is the stronger, more reliable signal; embedding fills gaps
        # for paraphrased/semantically-related but non-literal matches.
        combined = max(keyword_score, 0.6 * keyword_score + 0.4 * embedding_score, embedding_score * 0.9)
        results.append({
            "skill": _display_name(skill),
            "score": round(combined, 3),
        })
    return results


def calculate_match_score(resume: dict, jd: dict) -> dict:
    model = _get_model()

    resume_chunks = _resume_corpus(resume)
    if not resume_chunks:
        raise ValueError("Resume has no extractable content to score against.")
    resume_chunk_embeddings = model.encode(resume_chunks, convert_to_tensor=True)
    full_text_lower = " ".join(resume_chunks).lower()

    required_skills = jd.get("required_skills", [])
    preferred_skills = jd.get("preferred_skills", [])

    required_scores = _skill_match_scores(required_skills, resume_chunk_embeddings, resume_chunks, full_text_lower, model)
    preferred_scores = _skill_match_scores(preferred_skills, resume_chunk_embeddings, resume_chunks, full_text_lower, model)

    # Required skills weighted fully, preferred skills weighted at half importance
    all_weighted = [(s, 1.0) for s in required_scores] + [(s, 0.5) for s in preferred_scores]
    if all_weighted:
        weighted_sum = sum(s["score"] * w for s, w in all_weighted)
        weight_total = sum(w for _, w in all_weighted)
        skills_score = (weighted_sum / weight_total) * 100
    else:
        skills_score = 0.0

    matched_skills = [s for s in required_scores + preferred_scores if s["score"] >= SKILL_MATCH_THRESHOLD]
    missing_skills = [s for s in required_scores + preferred_scores if s["score"] < SKILL_MATCH_THRESHOLD]
    matched_skills.sort(key=lambda x: -x["score"])
    missing_skills.sort(key=lambda x: -x["score"])

    # Experience score: similarity between JD responsibilities and resume experience/projects text
    responsibilities = jd.get("responsibilities", [])
    exp_project_chunks = [
        c for c in resume_chunks
        if c not in (resume.get("skills", []) if isinstance(resume.get("skills"), list) else [])
    ]
    if responsibilities and exp_project_chunks:
        resp_text = " ".join(responsibilities)
        resp_embedding = model.encode(resp_text, convert_to_tensor=True)
        exp_embeddings = model.encode(exp_project_chunks, convert_to_tensor=True)
        experience_score = float(cos_sim(resp_embedding, exp_embeddings).max()) * 100
    else:
        experience_score = 0.0

    # Education score: simple semantic check between required and held education
    education_required = jd.get("education_required", "")
    resume_education = resume.get("education", [])
    if education_required and resume_education:
        edu_text = " ".join(json_safe_str(e) for e in resume_education)
        req_embedding = model.encode(education_required, convert_to_tensor=True)
        held_embedding = model.encode(edu_text, convert_to_tensor=True)
        education_score = float(cos_sim(req_embedding, held_embedding)) * 100
    else:
        education_score = 50.0  # neutral if nothing to compare

    overall_match = (
        skills_score * WEIGHT_SKILLS
        + experience_score * WEIGHT_EXPERIENCE
        + education_score * WEIGHT_EDUCATION
    )

    if overall_match >= 75:
        verdict = "High chance"
    elif overall_match >= 50:
        verdict = "Medium chance"
    else:
        verdict = "Low chance"

    return {
        "overall_match": round(overall_match, 1),
        "skills_score": round(skills_score, 1),
        "experience_score": round(experience_score, 1),
        "education_score": round(education_score, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys
    import json
    from agents.resume_parser import parse_resume
    from agents.job_analyzer import analyze_job_description_from_file

    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"

    resume = parse_resume(resume_path)
    jd = analyze_job_description_from_file(jd_path)
    result = calculate_match_score(resume, jd)
    print(json.dumps(result, indent=2))