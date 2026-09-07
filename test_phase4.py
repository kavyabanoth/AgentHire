"""
Phase 4 smoke test: Resume Parser + Job Analyzer + Match Score + Resume
Tailoring Agent, wired into the full pipeline so far.

Usage:
    python test_phase4.py "data/kavya_resume (1).pdf" data/sample_jd.txt kavya

Note: user_id must match the user_id used when the resume was ingested into
the Career Vault in Phase 1 (test_phase1.py used "kavya" by default).
"""

import sys
import json
from agents.resume_parser import parse_resume
from agents.job_analyzer import analyze_job_description_from_file
from agents.match_score import calculate_match_score
from agents.resume_tailor import tailor_resume


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "kavya"

    print(f"[1/4] Parsing resume: {resume_path}")
    resume = parse_resume(resume_path)

    print(f"[2/4] Analyzing job description: {jd_path}")
    jd = analyze_job_description_from_file(jd_path)

    print("[3/4] Calculating match score")
    match_result = calculate_match_score(resume, jd)
    print(f"  Overall match before tailoring: {match_result['overall_match']}% ({match_result['verdict']})\n")

    print("[4/4] Tailoring resume (grounded on Career Vault retrieval)\n")
    tailored = tailor_resume(resume, jd, match_result, user_id=user_id)

    print("=" * 60)
    print("TAILORED SUMMARY")
    print("=" * 60)
    print(tailored.get("summary", ""), "\n")

    print("=" * 60)
    print("REORDERED SKILLS")
    print("=" * 60)
    skills = tailored.get("skills", [])
    print(", ".join(skills) if isinstance(skills, list) else skills, "\n")

    print("=" * 60)
    print("TAILORING NOTES")
    print("=" * 60)
    for note in tailored.get("tailoring_notes", []):
        print(f"- {note}")

    print("\n" + "=" * 60)
    print("FULL TAILORED RESUME (JSON)")
    print("=" * 60)
    print(json.dumps(tailored, indent=2))


if __name__ == "__main__":
    main()