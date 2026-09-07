import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from environment (.env locally) first; fall back to Streamlit
    Cloud's st.secrets when deployed there. Safe to call outside Streamlit too."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-120b"  # free via Groq (llama-3.3-70b-versatile was deprecated)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, local, no API cost

CHROMA_PERSIST_DIR = "./data/chroma_store"
CAREER_VAULT_COLLECTION = "career_vault"

# Chunking config for resume sections
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50