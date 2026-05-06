
from src.validation.validator import validate_response


def hallucination_score(answer: str, context: str) -> dict:
    result = validate_response(answer, context)
    risk_map = {"low": 0.15, "medium": 0.5, "high": 0.85}
    return {
        "score": risk_map.get(result["hallucination_risk"], 0.5),
        "risk": result["hallucination_risk"],
        "details": result,
    }
