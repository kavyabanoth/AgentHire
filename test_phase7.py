"""
Phase 7 test: runs the ENTIRE AgentHire AI pipeline through the LangGraph
orchestrator in one call, instead of manually chaining agent functions like
test_phase1.py through test_phase6.py did.

Usage:
    python test_phase7.py "data/kavya_resume (1).pdf" data/sample_jd.txt kavya "TechCorp"
"""

import sys
import json
from orchestrator import run_pipeline


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "kavya"
    company_name = sys.argv[4] if len(sys.argv) > 4 else "your company"

    print("Running AgentHire AI pipeline via LangGraph orchestrator...\n")
    final_state = run_pipeline(resume_path, jd_path, user_id, company_name)

    print("=" * 60)
    print("PIPELINE EXECUTION LOG")
    print("=" * 60)
    for entry in final_state["log"]:
        print(entry)

    print("\n" + "=" * 60)
    print("MATCH SCORE")
    print("=" * 60)
    mr = final_state["match_result"]
    print(f"Overall: {mr['overall_match']}% ({mr['verdict']})")
    print(f"  Skills: {mr['skills_score']}% | Experience: {mr['experience_score']}% | Education: {mr['education_score']}%")

    print("\n" + "=" * 60)
    print("ATS SCORE")
    print("=" * 60)
    ats = final_state["ats_result"]
    print(f"Overall: {ats['ats_score']}/100 (Keyword coverage: {ats['keyword_coverage_pct']}%)")

    print("\n" + "=" * 60)
    print("TAILORED RESUME SUMMARY")
    print("=" * 60)
    print(final_state["tailored_resume"].get("summary", ""))

    print("\n" + "=" * 60)
    print("COVER LETTER")
    print("=" * 60)
    print(final_state["cover_letter_result"].get("cover_letter", ""))


if __name__ == "__main__":
    main()