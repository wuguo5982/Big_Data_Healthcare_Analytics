"""Utility functions for the NIH ClinicalBERT multimodal demo."""

from pathlib import Path
import json

import pandas as pd
import yaml


def project_root() -> Path:
    """Return the project root folder."""

    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load project configuration from configs/config.yaml."""

    config_path = project_root() / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict | list, output_path: str | Path) -> None:
    """Save a Python object as a formatted JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_ehr() -> pd.DataFrame:
    """Load structured EHR data."""

    ehr_path = project_root() / "data" / "processed" / "structured_ehr_1000.csv"

    return pd.read_csv(ehr_path)


def load_documents() -> pd.DataFrame:
    """Load clinical document data."""

    documents_path = (
        project_root()
        / "data"
        / "processed"
        / "clinical_documents_3000.csv"
    )

    return pd.read_csv(documents_path)


def build_evidence(patient_row: pd.Series) -> list[str]:
    """
    Build a simple list of clinical evidence for one patient.

    These rules are only for explainability in this research demo.
    They are not medical diagnosis rules.
    """

    evidence = []

    if patient_row.get("bnp", 0) > 450:
        evidence.append(f"BNP elevated at {patient_row['bnp']}.")

    if patient_row.get("lvef", 100) < 45:
        evidence.append(f"LVEF reduced at {patient_row['lvef']}%.")

    if patient_row.get("spo2", 100) < 92:
        evidence.append(f"Low oxygen saturation at {patient_row['spo2']}%.")

    if patient_row.get("wbc", 0) > 11:
        evidence.append(f"WBC elevated at {patient_row['wbc']}.")

    if patient_row.get("abnormal_cxr_label", 0) == 1:
        evidence.append("NIH ChestX-ray14 label indicates abnormal imaging.")

    if not evidence:
        evidence.append("No major high-risk evidence identified.")

    return evidence


def get_patient_documents(
    documents: pd.DataFrame,
    patient_id: str,
    max_documents: int = 5,
) -> list[str]:
    """Return a few representative clinical notes for one patient."""

    patient_documents = documents.loc[
        documents["patient_id"] == patient_id,
        "text",
    ]

    return patient_documents.head(max_documents).astype(str).tolist()


def get_risk_level(high_risk_label: int) -> str:
    """Convert numeric risk label into readable text."""

    if high_risk_label == 1:
        return "High"

    return "Low-to-Moderate"


def build_patient_summary(patient_id: str) -> dict:
    """Build an explainable summary for one patient."""

    ehr_data = load_ehr()
    documents = load_documents()

    patient_rows = ehr_data[ehr_data["patient_id"] == patient_id]

    if patient_rows.empty:
        raise ValueError(f"Patient {patient_id} not found.")

    patient_row = patient_rows.iloc[0]

    return {
        "patient_id": patient_row["patient_id"],
        "encounter_id": patient_row["encounter_id"],
        "image_id": patient_row["image_id"],
        "risk_level": get_risk_level(patient_row["high_risk_label"]),
        "primary_condition": patient_row["primary_condition"],
        "supporting_evidence": build_evidence(patient_row),
        "representative_documents": get_patient_documents(
            documents=documents,
            patient_id=patient_id,
        ),
        "disclaimer": "Research demo only. Not for clinical use.",
    }