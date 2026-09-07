"""
Phase 3 smoke test: Resume Parser + Job Analyzer + Match Score Agent, wired
into one end-to-end scoring pipeline.

Usage:
    python test_phase3.py "data/kavya_resume (1).pdf" data/sample_jd.txt
"""

import sys
import json
from agents.resume_parser import parse_resume
from agents.job_analyzer import analyze_job_description_from_file
from agents.match_score import calculate_match_score


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"

    print(f"[1/3] Parsing resume: {resume_path}")
    resume = parse_resume(resume_path)

    print(f"[2/3] Analyzing job description: {jd_path}")
    jd = analyze_job_description_from_file(jd_path)

    print("[3/3] Calculating match score\n")
    result = calculate_match_score(resume, jd)

    print(f"Overall Match: {result['overall_match']}%  ({result['verdict']})")
    print(f"  Skills Score:     {result['skills_score']}%")
    print(f"  Experience Score: {result['experience_score']}%")
    print(f"  Education Score:  {result['education_score']}%\n")

    print(f"Matched Skills ({len(result['matched_skills'])}):")
    for s in result["matched_skills"]:
        print(f"  ✓ {s['skill']}  (score: {s['score']})")

    print(f"\nMissing Skills ({len(result['missing_skills'])}):")
    for s in result["missing_skills"]:
        print(f"  ✗ {s['skill']}  (score: {s['score']})")


if __name__ == "__main__":
    main()