
"""
Lightweight LLM-as-judge style evaluator.

For local demo, this is deterministic. In production, this can call GPT-4o-mini
again with a judge prompt or use RAGAS/DeepEval/TruLens.
"""

from typing import Dict


def evaluate_response(answer: str, context: str) -> Dict[str, object]:
    answer_lower = answer.lower()
    context_lower = context.lower()

    has_evidence_word = any(w in answer_lower for w in ["evidence", "context", "cms", "audit", "billing"])
    has_limitation = any(w in answer_lower for w in ["manual review", "not a final", "insufficient", "limitation"])
    context_overlap = sum(1 for w in set(answer_lower.split()) if len(w) > 5 and w in context_lower)

    faithfulness = min(1.0, 0.45 + 0.05 * context_overlap + (0.2 if has_evidence_word else 0) + (0.15 if has_limitation else 0))
    groundedness = min(1.0, 0.4 + 0.06 * context_overlap)

    risk = "low" if faithfulness >= 0.75 and groundedness >= 0.65 else "medium"

    return {
        "faithfulness": round(faithfulness, 2),
        "groundedness": round(groundedness, 2),
        "hallucination_risk": risk,
        "judge_note": "Deterministic local judge; replace with GPT-4o-mini judge or RAGAS for production.",
    }
