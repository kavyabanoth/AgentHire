"""
ATS Optimizer Agent
---------------------
Responsibility: score a resume the way a real Applicant Tracking System
would -- via literal keyword parsing and structural checks, NOT semantic
similarity (that's what Phase 3's Match Score Agent already does).

Real ATS software (Workday, Greenhouse, Taleo, etc.) largely does:
- exact/fuzzy keyword string matching against the JD
- section detection (does it find "Experience", "Education", "Skills"?)
- basic formatting sanity (no parsing failures from tables/columns/images)
It does NOT understand that "ML" means "machine learning" unless both
appear literally, and it does NOT do embedding-based reasoning. Mirroring
that here is intentional and is the whole point of this agent existing
separately from the Match Score Agent.

Output:
    {
        "ats_score": float,               # 0-100
        "keyword_coverage_pct": float,
        "matched_keywords": [str],
        "missing_keywords": [str],
        "bullet_quality_score": float,     # 0-100
        "section_completeness_score": float,
        "formatting_issues": [str],
        "weak_bullets": [{"text": str, "issues": [str]}],
        "suggestions": [str]
    }
"""

import re
import json
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "led", "managed",
    "analyzed", "created", "optimized", "automated", "reduced", "increased",
    "achieved", "delivered", "launched", "engineered", "streamlined",
    "improved", "deployed", "architected", "spearheaded", "collaborated",
    "integrated", "conducted", "applied", "wrote", "trained", "identified",
    "resolved", "generated", "processed", "researched", "coordinated",
}

WEAK_PHRASES = [
    "responsible for", "worked on", "helped with", "helped ",
    "assisted", "involved in", "tasked with", "duties included",
]

REQUIRED_SECTIONS = ["summary", "skills", "experience", "education"]


def _flatten_bullets(resume: dict) -> list[str]:
    """Collect every bullet from experience and projects."""
    bullets = []
    for exp in resume.get("experience", []):
        if isinstance(exp, dict):
            bullets.extend(exp.get("bullets", []))
        else:
            bullets.append(str(exp))
    for proj in resume.get("projects", []):
        if isinstance(proj, dict):
            bullets.extend(proj.get("bullets", []))
    return [b for b in bullets if b.strip()]


def _bullet_quality(bullets: list[str]) -> tuple[float, list[dict]]:
    """Score bullets on action-verb start + quantified metrics.
    Returns (0-100 average score, list of weak bullets with issues)."""
    if not bullets:
        return 0.0, []

    total = 0.0
    weak = []
    for bullet in bullets:
        score = 0.0
        issues = []
        first_word = re.match(r"[A-Za-z]+", bullet.strip())
        first_word = first_word.group(0).lower() if first_word else ""

        if first_word in ACTION_VERBS:
            score += 0.5
        else:
            issues.append("Doesn't start with a strong action verb")

        if re.search(r"\d", bullet):
            score += 0.5
        else:
            issues.append("No quantifiable metric (number/%/count)")

        bullet_lower = bullet.lower()
        for phrase in WEAK_PHRASES:
            if phrase in bullet_lower:
                issues.append(f"Contains weak phrase: '{phrase.strip()}'")
                score = max(0.0, score - 0.2)
                break

        total += score
        if issues:
            weak.append({"text": bullet, "issues": issues})

    return round((total / len(bullets)) * 100, 1), weak


def _section_completeness(resume: dict) -> tuple[float, list[str]]:
    present = 0
    issues = []
    for section in REQUIRED_SECTIONS:
        val = resume.get(section)
        if val and (not isinstance(val, (list, str)) or len(val) > 0):
            present += 1
        else:
            issues.append(f"Missing or empty section: '{section}' -- ATS systems and recruiters expect this")
    return round((present / len(REQUIRED_SECTIONS)) * 100, 1), issues


def _extract_keyword_terms(jd: dict) -> list[str]:
    """Gather every JD term that matters for literal ATS keyword scanning."""
    terms = []
    for key in ("required_skills", "preferred_skills", "technologies", "keywords"):
        terms.extend(jd.get(key, []))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        t_clean = t.strip()
        if t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            unique.append(t_clean)
    return unique


def _keyword_coverage(resume: dict, jd_terms: list[str]) -> tuple[float, list[str], list[str]]:
    """Literal substring/word check -- this is how real ATS keyword scanners work."""
    resume_text_parts = [json.dumps(resume)]  # crude but literal, matches how ATS text-extracts everything
    full_text_lower = " ".join(resume_text_parts).lower()

    matched, missing = [], []
    for term in jd_terms:
        term_lower = re.sub(r"\s*\(.*?\)", "", term).strip().lower()  # drop parenthetical for the literal check
        if term_lower and term_lower in full_text_lower:
            matched.append(term)
        else:
            missing.append(term)

    coverage_pct = round((len(matched) / len(jd_terms)) * 100, 1) if jd_terms else 0.0
    return coverage_pct, matched, missing


def _generate_suggestions(missing_keywords: list[str], weak_bullets: list[dict], jd_role: str) -> list[str]:
    """LLM call for a short, concrete list of actionable suggestions -- not a rewrite,
    just prioritized advice the candidate can act on."""
    if not missing_keywords and not weak_bullets:
        return ["Resume already has strong ATS keyword coverage and bullet quality for this role."]

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
    prompt = f"""You are an ATS resume optimization advisor for a "{jd_role}" role.

Missing keywords (not found literally in the resume): {missing_keywords[:10]}
Weak bullets needing improvement: {json.dumps(weak_bullets[:5])}

Give exactly 3-5 short, concrete, actionable suggestions (one sentence each) for
how the candidate could improve ATS match -- e.g. which keyword to naturally work
into which section, or how to strengthen a specific bullet. Do NOT suggest fabricating
experience. Return ONLY a JSON list of strings, no markdown, no commentary."""

    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [content]  # fall back to raw text if not valid JSON


def optimize_ats(resume: dict, jd: dict) -> dict:
    bullets = _flatten_bullets(resume)
    bullet_quality_score, weak_bullets = _bullet_quality(bullets)
    section_completeness_score, section_issues = _section_completeness(resume)

    jd_terms = _extract_keyword_terms(jd)
    keyword_coverage_pct, matched_keywords, missing_keywords = _keyword_coverage(resume, jd_terms)

    ats_score = round(
        keyword_coverage_pct * 0.5
        + bullet_quality_score * 0.3
        + section_completeness_score * 0.2,
        1,
    )

    formatting_issues = list(section_issues)
    if not bullets:
        formatting_issues.append("No bullet points detected in experience/projects -- ATS parsers rely on these")

    suggestions = _generate_suggestions(missing_keywords, weak_bullets, jd.get("role_title", "this role"))

    return {
        "ats_score": ats_score,
        "keyword_coverage_pct": keyword_coverage_pct,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "bullet_quality_score": bullet_quality_score,
        "section_completeness_score": section_completeness_score,
        "formatting_issues": formatting_issues,
        "weak_bullets": weak_bullets,
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    import sys
    from agents.resume_parser import parse_resume
    from agents.job_analyzer import analyze_job_description_from_file

    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"

    resume = parse_resume(resume_path)
    jd = analyze_job_description_from_file(jd_path)
    result = optimize_ats(resume, jd)
    print(json.dumps(result, indent=2))