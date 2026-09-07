"""
Phase 5 smoke test: adds the ATS Optimizer Agent on top of the Phase 1-4
pipeline. Runs ATS scoring on the TAILORED resume (Phase 4 output), since
that's the version that will actually get submitted.

Usage:
    python test_phase5.py "data/kavya_resume (1).pdf" data/sample_jd.txt kavya
"""

import sys
import json
from agents.resume_parser import parse_resume
from agents.job_analyzer import analyze_job_description_from_file
from agents.match_score import calculate_match_score
from agents.resume_tailor import tailor_resume
from agents.ats_optimizer import optimize_ats


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "kavya"

    print(f"[1/5] Parsing resume: {resume_path}")
    resume = parse_resume(resume_path)

    print(f"[2/5] Analyzing job description: {jd_path}")
    jd = analyze_job_description_from_file(jd_path)

    print("[3/5] Calculating match score")
    match_result = calculate_match_score(resume, jd)

    print("[4/5] Tailoring resume")
    tailored = tailor_resume(resume, jd, match_result, user_id=user_id)

    print("[5/5] Running ATS optimization check on tailored resume\n")
    ats_result = optimize_ats(tailored, jd)

    print("=" * 60)
    print(f"ATS SCORE: {ats_result['ats_score']}/100")
    print("=" * 60)
    print(f"  Keyword Coverage:      {ats_result['keyword_coverage_pct']}%")
    print(f"  Bullet Quality:        {ats_result['bullet_quality_score']}%")
    print(f"  Section Completeness:  {ats_result['section_completeness_score']}%\n")

    print(f"Matched Keywords ({len(ats_result['matched_keywords'])}):")
    print(f"  {', '.join(ats_result['matched_keywords'])}\n")

    print(f"Missing Keywords ({len(ats_result['missing_keywords'])}):")
    print(f"  {', '.join(ats_result['missing_keywords'])}\n")

    if ats_result["formatting_issues"]:
        print("Formatting Issues:")
        for issue in ats_result["formatting_issues"]:
            print(f"  ⚠ {issue}")
        print()

    if ats_result["weak_bullets"]:
        print(f"Weak Bullets ({len(ats_result['weak_bullets'])}):")
        for wb in ats_result["weak_bullets"]:
            print(f"  - \"{wb['text'][:80]}...\"")
            for issue in wb["issues"]:
                print(f"      → {issue}")
        print()

    print("Suggestions:")
    for s in ats_result["suggestions"]:
        print(f"  • {s}")


if __name__ == "__main__":
    main()