"""
Phase 6 smoke test: adds the Cover Letter Agent on top of the full pipeline.

Usage:
    python test_phase6.py "data/kavya_resume (1).pdf" data/sample_jd.txt kavya "TechCorp"
"""

import sys
import json
from agents.resume_parser import parse_resume
from agents.job_analyzer import analyze_job_description_from_file
from agents.match_score import calculate_match_score
from agents.cover_letter import generate_cover_letter


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "kavya"
    company_name = sys.argv[4] if len(sys.argv) > 4 else "your company"

    print(f"[1/4] Parsing resume: {resume_path}")
    resume = parse_resume(resume_path)

    print(f"[2/4] Analyzing job description: {jd_path}")
    jd = analyze_job_description_from_file(jd_path)

    print("[3/4] Calculating match score")
    match_result = calculate_match_score(resume, jd)

    print(f"[4/4] Generating cover letter for {company_name}\n")
    result = generate_cover_letter(resume, jd, match_result, company_name=company_name, user_id=user_id)

    print("=" * 60)
    print("COVER LETTER")
    print("=" * 60)
    print(result.get("cover_letter", ""), "\n")

    print("=" * 60)
    print("KEY POINTS REFERENCED (for verification against your real resume)")
    print("=" * 60)
    for point in result.get("key_points_used", []):
        print(f"  ✓ {point}")


if __name__ == "__main__":
    main()