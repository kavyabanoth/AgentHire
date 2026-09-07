"""
Phase 2 smoke test: Job Analyzer Agent.

Usage:
    python test_phase2.py path/to/job_description.txt

If a career vault already exists from Phase 1 (test_phase1.py run), this also
does a quick cross-check: for each required skill in the JD, query the vault
to see if related experience exists -- a preview of what the Match Score
Agent (Phase 3) will formalize.
"""

import sys
import json
from agents.job_analyzer import analyze_job_description_from_file
from career_vault.ingest import query_career_vault


def main():
    jd_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_jd.txt"

    print(f"[1/2] Analyzing job description: {jd_path}")
    jd = analyze_job_description_from_file(jd_path)
    print(json.dumps(jd, indent=2), "\n")

    print("[2/2] Cross-checking required skills against Career Vault (if populated)")
    required_skills = jd.get("required_skills", [])
    if not required_skills:
        print("No required_skills extracted -- skipping cross-check.")
        return

    for skill in required_skills[:5]:  # cap to first 5 for a quick preview
        try:
            hits = query_career_vault(skill, user_id="kavya", n_results=1)
            docs = hits.get("documents", [[]])[0]
            if docs:
                print(f"- '{skill}' -> closest match: {docs[0][:100]}...")
            else:
                print(f"- '{skill}' -> no match found in Career Vault")
        except Exception as e:
            print(f"- '{skill}' -> Career Vault query failed ({e}); run test_phase1.py first")
            break


if __name__ == "__main__":
    main()