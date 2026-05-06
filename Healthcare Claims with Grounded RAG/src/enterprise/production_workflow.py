
"""
Production-style Production enterprise workflow.

Flow:
User Query
-> Local RAG Retriever
-> Reranker
-> Grounded Context
-> GPT-4o-mini Reasoning
-> Validation
-> Hallucination Score
-> Judge Evaluation
-> UI-ready payload
"""

from __future__ import annotations

from typing import Dict, Any
from src.rag.retriever import retrieve_context
from src.enterprise.reranker import rerank_context
from src.llm.gpt4omini_engine import generate_grounded_response, is_openai_ready
from src.validation.validator import validate_response
from src.hallucination.scorer import hallucination_score
from src.enterprise.judge import evaluate_response


def run_production_analysis(query: str, top_k: int = 3, use_gpt4omini: bool = True) -> Dict[str, Any]:
    """Run the full Production workflow and return a UI-ready dictionary."""
    query = (query or "").strip()

    if not query:
        return {
            "status": "error",
            "message": "Please enter a healthcare FWA question.",
            "query": query,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "grounded_context": "",
            "answer": "",
            "validation": {},
            "hallucination": {},
            "judge": {},
            "openai_ready": is_openai_ready(),
        }

    retrieved_chunks = retrieve_context(query, top_k=max(top_k, 1))
    reranked_chunks = rerank_context(query, retrieved_chunks, top_k=top_k)
    grounded_context = "\n\n---\n\n".join(reranked_chunks)

    if use_gpt4omini:
        answer = generate_grounded_response(query, grounded_context)
    else:
        answer = (
            "Evidence Used\n"
            f"- GPT-4o-mini was disabled in the UI. Retrieved context length: {len(grounded_context)} characters.\n\n"
            "Risk Reasoning\n"
            "- Review the retrieved policy/audit context for abnormal billing, duplicate claims, diagnosis-procedure inconsistency, and provider outlier behavior.\n\n"
            "Hallucination / Evidence Limitation\n"
            "- No LLM reasoning was performed.\n\n"
            "Recommended Next Step\n"
            "- Enable GPT-4o-mini for grounded reasoning or escalate to manual review."
        )

    validation = validate_response(answer, grounded_context)
    hallucination = hallucination_score(answer, grounded_context)
    judge = evaluate_response(answer, grounded_context)

    return {
        "status": "success",
        "message": "Production analysis completed.",
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": reranked_chunks,
        "grounded_context": grounded_context,
        "answer": answer,
        "validation": validation,
        "hallucination": hallucination,
        "judge": judge,
        "openai_ready": is_openai_ready(),
    }
