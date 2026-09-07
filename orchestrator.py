"""
AgentHire AI -- LangGraph Orchestrator
-----------------------------------------
Wires the 6 agents built in Phases 1-6 into a single StateGraph. Each agent
becomes a node; a shared AgentState flows through the graph, accumulating
results as it goes. This replaces manually chaining function calls in each
test_phaseN.py script with a real orchestrated pipeline.

Graph flow:

    START
      |
    parse_resume
      |
    ingest_career_vault
      |
    analyze_job_description
      |
    calculate_match_score
      |
    tailor_resume
      |
    optimize_ats
      |
    generate_cover_letter
      |
     END

Each node reads what it needs from AgentState and writes its output back
into it -- LangGraph merges partial state updates automatically.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from agents.resume_parser import parse_resume
from career_vault.ingest import ingest_resume
from agents.job_analyzer import analyze_job_description_from_file
from agents.match_score import calculate_match_score
from agents.resume_tailor import tailor_resume
from agents.ats_optimizer import optimize_ats
from agents.cover_letter import generate_cover_letter


class AgentState(TypedDict, total=False):
    # --- inputs ---
    resume_path: str
    jd_path: str
    user_id: str
    company_name: str

    # --- accumulated outputs (filled in by nodes as the graph runs) ---
    resume: dict
    ingestion_result: dict
    jd: dict
    match_result: dict
    tailored_resume: dict
    ats_result: dict
    cover_letter_result: dict

    # --- run log for observability/demo purposes ---
    log: list


def _log(state: AgentState, message: str) -> list:
    existing = state.get("log", [])
    return existing + [message]


# ---- Node functions -------------------------------------------------------

def node_parse_resume(state: AgentState) -> dict:
    resume = parse_resume(state["resume_path"])
    return {"resume": resume, "log": _log(state, f"[1/6] Parsed resume: {state['resume_path']}")}


def node_ingest_career_vault(state: AgentState) -> dict:
    result = ingest_resume(state["resume"], user_id=state["user_id"])
    return {"ingestion_result": result, "log": _log(state, f"[2/6] Ingested {result['ingested_chunks']} chunks into Career Vault")}


def node_analyze_jd(state: AgentState) -> dict:
    jd = analyze_job_description_from_file(state["jd_path"])
    return {"jd": jd, "log": _log(state, f"[3/6] Analyzed JD: {jd.get('role_title', 'Unknown role')}")}


def node_match_score(state: AgentState) -> dict:
    result = calculate_match_score(state["resume"], state["jd"])
    return {"match_result": result, "log": _log(state, f"[4/6] Match score: {result['overall_match']}% ({result['verdict']})")}


def node_tailor_resume(state: AgentState) -> dict:
    tailored = tailor_resume(state["resume"], state["jd"], state["match_result"], user_id=state["user_id"])
    return {"tailored_resume": tailored, "log": _log(state, "[5/6] Resume tailored")}


def node_ats_optimize(state: AgentState) -> dict:
    result = optimize_ats(state["tailored_resume"], state["jd"])
    return {"ats_result": result, "log": _log(state, f"[6/6] ATS score: {result['ats_score']}/100")}


def node_cover_letter(state: AgentState) -> dict:
    result = generate_cover_letter(
        state["resume"],
        state["jd"],
        state["match_result"],
        company_name=state.get("company_name", "your company"),
        user_id=state["user_id"],
    )
    return {"cover_letter_result": result, "log": _log(state, "[7/7] Cover letter generated")}


# ---- Build the graph --------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_resume", node_parse_resume)
    graph.add_node("ingest_career_vault", node_ingest_career_vault)
    graph.add_node("analyze_jd", node_analyze_jd)
    graph.add_node("match_score", node_match_score)
    graph.add_node("tailor_resume", node_tailor_resume)
    graph.add_node("ats_optimize", node_ats_optimize)
    graph.add_node("cover_letter", node_cover_letter)

    graph.add_edge(START, "parse_resume")
    graph.add_edge("parse_resume", "ingest_career_vault")
    graph.add_edge("ingest_career_vault", "analyze_jd")
    graph.add_edge("analyze_jd", "match_score")
    graph.add_edge("match_score", "tailor_resume")
    graph.add_edge("tailor_resume", "ats_optimize")
    graph.add_edge("ats_optimize", "cover_letter")
    graph.add_edge("cover_letter", END)

    return graph.compile()


def run_pipeline(resume_path: str, jd_path: str, user_id: str = "default_user", company_name: str = "your company") -> AgentState:
    """Convenience entrypoint: run the full AgentHire AI pipeline end-to-end."""
    app = build_graph()
    initial_state: AgentState = {
        "resume_path": resume_path,
        "jd_path": jd_path,
        "user_id": user_id,
        "company_name": company_name,
        "log": [],
    }
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import sys
    import json

    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "kavya"
    company_name = sys.argv[4] if len(sys.argv) > 4 else "your company"

    final_state = run_pipeline(resume_path, jd_path, user_id, company_name)

    print("=" * 60)
    print("PIPELINE LOG")
    print("=" * 60)
    for entry in final_state["log"]:
        print(entry)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Match: {final_state['match_result']['overall_match']}% ({final_state['match_result']['verdict']})")
    print(f"ATS Score: {final_state['ats_result']['ats_score']}/100")
    print(f"\nCover Letter Preview:\n{final_state['cover_letter_result']['cover_letter'][:300]}...")