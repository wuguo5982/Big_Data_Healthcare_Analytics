
"""
Simple local RAG retriever.

For portfolio/demo use, this performs keyword scoring over CMS policy text and audit rules.
It can later be replaced with FAISS/Chroma/OpenSearch without changing the Production workflow.
"""

from pathlib import Path
import json
from typing import List
from src.config import POLICY_PATH, AUDIT_RULES_PATH


def _load_policy_text() -> str:
    if POLICY_PATH.exists():
        return POLICY_PATH.read_text(errors="ignore")
    return (
        "CMS FWA audit guidance: abnormal billing patterns, excessive utilization, "
        "diagnosis-procedure inconsistencies, duplicate claims, and outlier provider behavior "
        "may require manual audit review."
    )


def _load_audit_rules() -> str:
    if AUDIT_RULES_PATH.exists():
        try:
            data = json.loads(AUDIT_RULES_PATH.read_text(errors="ignore"))
            return json.dumps(data, indent=2)
        except Exception:
            return AUDIT_RULES_PATH.read_text(errors="ignore")
    return "Audit rules include high charge outliers, duplicate billing, and diagnosis-procedure mismatch."


def chunk_text(text: str, chunk_size: int = 700) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks or [text]


def retrieve_context(query: str, top_k: int = 4) -> List[str]:
    corpus = _load_policy_text() + "\n\n" + _load_audit_rules()
    chunks = chunk_text(corpus, chunk_size=120)

    query_terms = {t.lower().strip(".,:;()[]") for t in query.split() if len(t) > 2}

    scored = []
    for chunk in chunks:
        chunk_terms = set(chunk.lower().split())
        score = len(query_terms.intersection(chunk_terms))
        # light preference for important FWA terms
        for term in ["fraud", "waste", "abuse", "billing", "diagnosis", "procedure", "duplicate", "outlier", "audit"]:
            if term in chunk.lower():
                score += 0.25
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
