from __future__ import annotations
import re

def enforce_fwa_language(answer: str) -> str:
    banned = ["committed fraud", "is guilty", "criminal", "definitely fraud"]
    fixed = answer
    for b in banned:
        fixed = re.sub(b, "is flagged for potential FWA risk", fixed, flags=re.I)
    if "final fraud determination" not in fixed.lower():
        fixed += "\n\nCompliance note: This system flags potential FWA risk only. It is not a final fraud determination; human investigator review is required."
    return fixed

def validate_grounding(answer: str, citations: str) -> tuple[bool, str]:
    if not citations or citations.startswith("No policy"):
        return False, "Policy citations are missing. Answer should be treated as lower confidence."
    if len(answer.strip()) < 30:
        return False, "Answer is too short to be useful."
    return True, "Grounding check passed."
