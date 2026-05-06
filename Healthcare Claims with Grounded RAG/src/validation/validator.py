
"""
Validation layer for hallucination reduction.

The validator checks whether the response appears grounded in retrieved context
and whether it uses unsafe unsupported conclusion language.
"""

from typing import Dict


UNSAFE_FINAL_WORDS = [
    "definitely fraud",
    "certainly fraud",
    "must be denied",
    "automatically deny",
    "criminal fraud",
]


def validate_response(answer: str, context: str) -> Dict[str, object]:
    answer_lower = answer.lower()
    context_lower = context.lower()

    evidence_terms = [
        "billing", "claim", "diagnosis", "procedure", "audit", "cms",
        "duplicate", "outlier", "utilization", "manual review"
    ]
    supported_terms = [t for t in evidence_terms if t in answer_lower and t in context_lower]
    unsupported_risky = [w for w in UNSAFE_FINAL_WORDS if w in answer_lower]

    grounding_score = min(1.0, len(supported_terms) / 5)

    if unsupported_risky:
        hallucination_risk = "high"
    elif grounding_score >= 0.6:
        hallucination_risk = "low"
    elif grounding_score >= 0.3:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "medium"

    return {
        "grounding_score": round(grounding_score, 2),
        "hallucination_risk": hallucination_risk,
        "supported_terms": supported_terms,
        "unsupported_risky_phrases": unsupported_risky,
        "recommendation": "Manual review required; AI output is not a final fraud determination.",
    }
