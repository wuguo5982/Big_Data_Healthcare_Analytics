from __future__ import annotations
import pandas as pd

SUPPORTIVE_TERMS = ["supports", "symptoms", "assessment", "care plan", "oxygen", "mobility", "hospice"]
CONCERN_TERMS = ["ambulates independently", "no oxygen", "no documented", "may require additional", "routine follow-up"]

def note_support_score(note: str) -> float:
    text = (note or "").lower()
    score = 0.5
    score += 0.12 * sum(t in text for t in SUPPORTIVE_TERMS)
    score -= 0.18 * sum(t in text for t in CONCERN_TERMS)
    return max(0.0, min(1.0, score))

def analyze_claim_notes(claim_id: str, notes: pd.DataFrame) -> dict:
    subset = notes[notes["claim_id"].astype(str) == str(claim_id)]
    if subset.empty:
        return {"claim_id": claim_id, "support_score": None, "summary": "No doctor note found for this claim."}
    note = subset.iloc[0]["clinical_note"]
    score = note_support_score(note)
    if score >= .7: verdict = "Clinical documentation appears supportive."
    elif score >= .4: verdict = "Clinical documentation is partially supportive; manual review recommended."
    else: verdict = "Clinical documentation may not support the billed service."
    return {"claim_id": claim_id, "support_score": round(score, 3), "summary": verdict, "note_excerpt": note[:600]}
