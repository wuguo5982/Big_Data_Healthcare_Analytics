
"""
Lightweight reranker.

This is intentionally simple for low/medium local compute.
You can later replace it with a cross-encoder reranker such as BGE reranker.
"""

from typing import List


def rerank_context(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    q_terms = {t.lower().strip(".,:;()[]") for t in query.split() if len(t) > 2}

    def score(chunk: str) -> float:
        c = chunk.lower()
        overlap = sum(1 for t in q_terms if t in c)
        domain_boost = sum(1 for t in ["cms", "audit", "billing", "diagnosis", "procedure", "claim", "fraud"] if t in c)
        return overlap + 0.5 * domain_boost

    return sorted(chunks, key=score, reverse=True)[:top_k]
