"""Document and RAG analytics utilities for the Streamlit dashboard.

This module keeps analytics lightweight and local. It summarizes structured
claims, unstructured clinical notes, RAG corpus quality, retrieval quality,
grounding, validation, and hallucination-reduction signals.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import (
    AUDIT_RULES_PATH,
    CLAIMS_PATH,
    CLINICAL_NOTES_PATH,
    DIAGNOSIS_POLICY_PATH,
    POLICY_PATH,
    PROCEDURE_BENCHMARK_PATH,
)

STOPWORDS = {
    "the", "and", "or", "to", "of", "a", "in", "for", "is", "are", "be", "with",
    "should", "this", "that", "when", "by", "as", "an", "on", "not", "from", "it",
    "was", "were", "can", "may", "must", "review", "claim", "claims", "documentation",
    "synthetic", "procedure", "diagnosis", "provider", "patient",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text).lower()) if t not in STOPWORDS]


def corpus_sources() -> Dict[str, str]:
    """Return text sources used by the local RAG knowledge base."""
    rules_text = json.dumps(json.loads(Path(AUDIT_RULES_PATH).read_text(encoding="utf-8")), indent=2)
    return {
        "CMS-style policy document": Path(POLICY_PATH).read_text(encoding="utf-8"),
        "Procedure benchmark table": Path(PROCEDURE_BENCHMARK_PATH).read_text(encoding="utf-8"),
        "Diagnosis policy mapping": Path(DIAGNOSIS_POLICY_PATH).read_text(encoding="utf-8"),
        "Audit rules": rules_text,
        "Synthetic clinical notes": "\n".join(pd.read_csv(CLINICAL_NOTES_PATH)["note_text"].astype(str).tolist()),
    }


def document_inventory() -> pd.DataFrame:
    """Summarize the RAG/document corpus included in the project."""
    rows = []
    for name, text in corpus_sources().items():
        tokens = tokenize(text)
        approx_chunks = max(1, len(re.split(r"\n\s*\n", text)))
        rows.append(
            {
                "source": name,
                "characters": len(text),
                "tokens_approx": len(tokens),
                "unique_terms": len(set(tokens)),
                "approx_chunks_or_rows": approx_chunks,
            }
        )
    return pd.DataFrame(rows)


def top_terms(n: int = 20) -> pd.DataFrame:
    """Return top corpus terms for quick document analytics."""
    counter: Counter[str] = Counter()
    for text in corpus_sources().values():
        counter.update(tokenize(text))
    return pd.DataFrame(counter.most_common(n), columns=["term", "count"])


def claims_summary() -> Dict[str, Any]:
    df = pd.read_csv(CLAIMS_PATH)
    return {
        "rows": int(len(df)),
        "labeled_fwa_rate": round(float(df["label"].mean()), 3),
        "avg_claim_amount": round(float(df["claim_amount"].mean()), 2),
        "median_claim_amount": round(float(df["claim_amount"].median()), 2),
        "high_risk_provider_rate": round(float(df["is_high_risk_provider"].mean()), 3),
        "procedure_count": int(df["procedure_code"].nunique()),
        "diagnosis_count": int(df["diagnosis_code"].nunique()),
        "provider_count": int(df["provider_id"].nunique()),
    }


def validation_report() -> Dict[str, Any]:
    """Basic data validation for CI/CD and Streamlit visibility."""
    df = pd.read_csv(CLAIMS_PATH)
    notes = pd.read_csv(CLINICAL_NOTES_PATH)
    proc = pd.read_csv(PROCEDURE_BENCHMARK_PATH)
    diag = pd.read_csv(DIAGNOSIS_POLICY_PATH)

    required_claim_cols = {
        "claim_id", "provider_id", "patient_id", "procedure_code", "diagnosis_code",
        "claim_amount", "patient_age", "num_prior_claims", "days_since_last_claim",
        "is_high_risk_provider", "label",
    }
    missing_cols = sorted(required_claim_cols - set(df.columns))
    orphan_note_claims = sorted(set(notes["claim_id"]) - set(df["claim_id"]))
    unknown_proc = sorted(set(df["procedure_code"].astype(str)) - set(proc["procedure_code"].astype(str)))
    unknown_diag = sorted(set(df["diagnosis_code"].astype(str)) - set(diag["diagnosis_code"].astype(str)))

    return {
        "schema_valid": not missing_cols,
        "missing_claim_columns": missing_cols,
        "claim_rows": int(len(df)),
        "clinical_note_rows": int(len(notes)),
        "one_note_per_claim": int(len(notes)) == int(len(df)),
        "orphan_note_claim_count": len(orphan_note_claims),
        "unknown_procedure_codes": unknown_proc,
        "unknown_diagnosis_codes": unknown_diag,
        "null_counts": {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0},
        "amount_positive": bool((df["claim_amount"] > 0).all()),
    }


def retrieval_quality(rag_engine, sample_queries: List[str] | None = None) -> pd.DataFrame:
    """Evaluate whether the RAG index returns usable evidence for representative queries."""
    if sample_queries is None:
        sample_queries = [
            "upcoding high complexity office visit documentation medical necessity",
            "duplicate billing repeated claims within seven days",
            "durable medical equipment oxygen concentrator medical necessity",
            "unclassified drug code pricing support documentation",
            "provider outlier peer benchmark high utilization human review",
            "ambulance transport medical necessity origin destination",
        ]
    rows = []
    for q in sample_queries:
        hits = rag_engine.retrieve(q, top_k=4)
        scores = [float(h["score"]) for h in hits]
        rows.append(
            {
                "query": q,
                "top_score": round(max(scores), 3) if scores else 0.0,
                "avg_top4_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "chunks_returned": len(hits),
                "retrieval_gate_pass": bool(scores and max(scores) >= 0.05 and len(hits) >= 2),
            }
        )
    return pd.DataFrame(rows)
