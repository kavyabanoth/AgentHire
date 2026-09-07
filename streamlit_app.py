"""
AgentHire AI -- Streamlit Dashboard
--------------------------------------
UI wrapper around the Phase 7 LangGraph orchestrator. Upload a resume,
paste a job description, and view match score, tailored resume, ATS
report, and cover letter -- all in one place, with downloads.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile
import os
import json
from pathlib import Path
from orchestrator import run_pipeline

st.set_page_config(page_title="AgentHire AI", page_icon="🎯", layout="wide")

st.title("🎯 AgentHire AI")
st.caption("Multi-agent resume tailoring & job-match platform -- powered by LangGraph + Groq + ChromaDB")

with st.sidebar:
    st.header("Inputs")
    user_id = st.text_input("User ID", value="default_user", help="Namespaces your Career Vault entries in ChromaDB")
    company_name = st.text_input("Company Name", value="the company")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Paste Job Description", height=300, placeholder="Paste the full job description here...")
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    st.divider()
    st.caption("Pipeline: Parse → Ingest → Analyze JD → Match Score → Tailor → ATS Check → Cover Letter")

if run_button:
    if not resume_file:
        st.error("Please upload a resume file.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        os.makedirs("data", exist_ok=True)
        resume_path = jd_path = None
        with st.spinner("Running multi-agent pipeline... this can take 30-60 seconds"):
            try:
                suffix = Path(resume_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="data") as tmp_resume:
                    tmp_resume.write(resume_file.getvalue())
                    resume_path = tmp_resume.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8", dir="data") as tmp_jd:
                    tmp_jd.write(jd_text)
                    jd_path = tmp_jd.name

                final_state = run_pipeline(resume_path, jd_path, user_id=user_id, company_name=company_name)
                st.session_state["result"] = final_state
                st.success("Pipeline complete!")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
            finally:
                for p in (resume_path, jd_path):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

if "result" in st.session_state:
    state = st.session_state["result"]
    match = state["match_result"]
    ats = state["ats_result"]
    tailored = state["tailored_resume"]
    cover = state["cover_letter_result"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Match Score", "📝 Tailored Resume", "✅ ATS Report", "✉️ Cover Letter", "🔧 Pipeline Log"]
    )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Match", f"{match['overall_match']}%", match["verdict"])
        col2.metric("Skills", f"{match['skills_score']}%")
        col3.metric("Experience", f"{match['experience_score']}%")
        col4.metric("Education", f"{match['education_score']}%")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched Skills")
            for s in match["matched_skills"]:
                st.write(f"**{s['skill']}**  (score: {s['score']})")
        with c2:
            st.subheader("❌ Missing Skills")
            for s in match["missing_skills"]:
                st.write(f"**{s['skill']}**  (score: {s['score']})")

    with tab2:
        st.subheader("Tailored Summary")
        st.write(tailored.get("summary", ""))

        st.subheader("Reordered Skills")
        skills = tailored.get("skills", [])
        st.write(", ".join(skills) if isinstance(skills, list) else skills)

        st.subheader("Experience")
        for exp in tailored.get("experience", []):
            if isinstance(exp, dict):
                st.markdown(f"**{exp.get('title', '')} — {exp.get('company', '')}** ({exp.get('duration', '')})")
                for b in exp.get("bullets", []):
                    st.write(f"- {b}")

        st.subheader("Projects")
        for proj in tailored.get("projects", []):
            if isinstance(proj, dict):
                st.markdown(f"**{proj.get('title', '')}**")
                st.caption(proj.get("description", ""))
                for b in proj.get("bullets", []):
                    st.write(f"- {b}")

        st.subheader("What Was Changed")
        for note in tailored.get("tailoring_notes", []):
            st.write(f"- {note}")

        st.download_button(
            "⬇ Download Tailored Resume (JSON)",
            data=json.dumps(tailored, indent=2),
            file_name="tailored_resume.json",
            mime="application/json",
        )

    with tab3:
        col1, col2, col3 = st.columns(3)
        col1.metric("ATS Score", f"{ats['ats_score']}/100")
        col2.metric("Keyword Coverage", f"{ats['keyword_coverage_pct']}%")
        col3.metric("Bullet Quality", f"{ats['bullet_quality_score']}%")

        st.subheader("Missing Keywords")
        st.write(", ".join(ats["missing_keywords"]) if ats["missing_keywords"] else "None -- full coverage!")

        if ats["formatting_issues"]:
            st.subheader("Formatting Issues")
            for issue in ats["formatting_issues"]:
                st.warning(issue)

        if ats["weak_bullets"]:
            st.subheader("Weak Bullets")
            for wb in ats["weak_bullets"]:
                with st.expander(wb["text"][:80] + "..."):
                    for issue in wb["issues"]:
                        st.write(f"→ {issue}")

        st.subheader("💡 Suggestions")
        for s in ats["suggestions"]:
            st.info(s)

    with tab4:
        st.text_area("Cover Letter", value=cover.get("cover_letter", ""), height=400)
        st.download_button(
            "⬇ Download Cover Letter (.txt)",
            data=cover.get("cover_letter", ""),
            file_name="cover_letter.txt",
        )

        st.subheader("Key Points Referenced (verify against your real resume)")
        for p in cover.get("key_points_used", []):
            st.write(f"✓ {p}")

    with tab5:
        st.subheader("Execution Log")
        for entry in state["log"]:
            st.text(entry)

else:
    st.info("👈 Upload a resume and paste a job description in the sidebar, then click **Run Pipeline**.")