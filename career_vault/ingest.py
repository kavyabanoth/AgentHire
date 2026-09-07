"""
Career Vault Ingestion
-----------------------
Takes the structured resume dict (from agents/resume_parser.py) and stores
it in ChromaDB as retrievable chunks, tagged by section type. This is the
"grounding" store the Resume Tailoring Agent later retrieves from, so
generated bullets are always traceable back to real candidate history.

Each chunk is stored with metadata: {"section": ..., "source_title": ...}
so retrieval can be filtered (e.g. "only pull from projects" or
"only pull from experience at Company X").
"""

import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_PERSIST_DIR, CAREER_VAULT_COLLECTION, EMBEDDING_MODEL


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=CAREER_VAULT_COLLECTION,
        embedding_function=embed_fn,
    )


def _flatten_resume(resume: dict) -> list[dict]:
    """Convert structured resume dict into a flat list of {text, metadata} chunks.

    Tolerant of LLM output variance: experience/projects entries may come back
    as either structured dicts ({title, company, bullets}) or plain strings,
    depending on how strictly the model followed the extraction schema.
    """
    chunks = []

    if resume.get("summary"):
        chunks.append({
            "text": resume["summary"],
            "metadata": {"section": "summary", "source_title": "Summary"},
        })

    for exp in resume.get("experience", []):
        if isinstance(exp, dict):
            title = f"{exp.get('title', '')} at {exp.get('company', '')}".strip()
            bullets_text = "\n".join(exp.get("bullets", []))
            text = f"{title}\n{bullets_text}"
        else:
            # Plain string entry (model returned experience as a text block)
            title = str(exp)[:80]
            text = str(exp)
        chunks.append({
            "text": text,
            "metadata": {"section": "experience", "source_title": title},
        })

    for proj in resume.get("projects", []):
        if isinstance(proj, dict):
            title = proj.get("title", "")
            text = f"{title}\n{proj.get('description', '')}\nTech: {', '.join(proj.get('tech', []))}"
        else:
            title = str(proj)[:80]
            text = str(proj)
        chunks.append({
            "text": text,
            "metadata": {"section": "projects", "source_title": title},
        })

    if resume.get("skills"):
        skills = resume["skills"]
        skills_text = ", ".join(skills) if isinstance(skills, list) else str(skills)
        chunks.append({
            "text": skills_text,
            "metadata": {"section": "skills", "source_title": "Skills"},
        })

    for cert in resume.get("certifications", []):
        cert_text = cert if isinstance(cert, str) else str(cert)
        chunks.append({
            "text": cert_text,
            "metadata": {"section": "certifications", "source_title": cert_text[:80]},
        })

    for edu in resume.get("education", []):
        edu_text = edu if isinstance(edu, str) else str(edu)
        chunks.append({
            "text": edu_text,
            "metadata": {"section": "education", "source_title": edu_text[:80]},
        })

    return chunks


def ingest_resume(resume: dict, user_id: str = "default_user"):
    """Ingest a structured resume dict into the Career Vault collection."""
    collection = get_collection()
    chunks = _flatten_resume(resume)

    if not chunks:
        raise ValueError("No content extracted from resume to ingest.")

    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{user_id}_{chunk['metadata']['section']}_{i}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        meta = dict(chunk["metadata"])
        meta["user_id"] = user_id
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return {"ingested_chunks": len(ids), "user_id": user_id}


def query_career_vault(query_text: str, user_id: str = "default_user", n_results: int = 5, section: str | None = None):
    """Retrieve top-k relevant career vault chunks for a query (e.g. a JD requirement)."""
    collection = get_collection()
    where_filter = {"user_id": user_id}
    if section:
        where_filter["section"] = section

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter,
    )
    return results


if __name__ == "__main__":
    import json
    import sys
    from agents.resume_parser import parse_resume

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resume.pdf"
    resume = parse_resume(path)
    result = ingest_resume(resume)
    print(json.dumps(result, indent=2))