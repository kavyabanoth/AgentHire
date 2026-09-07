"""
Phase 1 smoke test: Resume Parser + Career Vault ingestion.

Usage:
    python test_phase1.py path/to/resume.pdf
"""

import sys
import json
from agents.resume_parser import parse_resume
from career_vault.ingest import ingest_resume, query_career_vault


def main():
    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"

    print(f"[1/3] Parsing resume: {resume_path}")
    resume = parse_resume(resume_path)
    print(json.dumps(resume, indent=2)[:800], "...\n")

    print("[2/3] Ingesting into Career Vault (ChromaDB)")
    result = ingest_resume(resume, user_id="kavya")
    print(result, "\n")

    print("[3/3] Test retrieval — querying for a sample JD requirement")
    test_query = "experience with Python and machine learning"
    hits = query_career_vault(test_query, user_id="kavya", n_results=3)
    for doc, meta in zip(hits["documents"][0], hits["metadatas"][0]):
        print(f"- [{meta['section']}] {doc[:120]}...")


if __name__ == "__main__":
    main()
