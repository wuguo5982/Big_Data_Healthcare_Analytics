"""Generate explainable JSON summaries for sample patients."""

import json

import pandas as pd

from utils import load_config, project_root, build_evidence


def get_risk_level(high_risk_label: int) -> str:
    """Convert numeric risk label into readable text."""

    if high_risk_label == 1:
        return "High"

    return "Low-to-Moderate"


def get_patient_documents(
    documents: pd.DataFrame,
    patient_id: str,
    max_documents: int = 5,
) -> list[str]:
    """Get a few representative clinical notes for one patient."""

    patient_documents = documents.loc[
        documents["patient_id"] == patient_id,
        "text",
    ]

    return patient_documents.head(max_documents).astype(str).tolist()


def create_patient_summary(
    patient_row: pd.Series,
    documents: pd.DataFrame,
) -> dict:
    """Create one explainable JSON summary for one patient."""

    patient_id = patient_row["patient_id"]

    return {
        "patient_id": patient_id,
        "encounter_id": patient_row["encounter_id"],
        "image_id": patient_row["image_id"],
        "risk_level": get_risk_level(patient_row["high_risk_label"]),
        "primary_condition": patient_row["primary_condition"],
        "supporting_evidence": build_evidence(patient_row),
        "representative_documents": get_patient_documents(
            documents=documents,
            patient_id=patient_id,
        ),
        "clinical_interpretation": (
            "This summary combines chest X-ray labels, structured EHR features, "
            "and ClinicalBERT-ready clinical text."
        ),
        "recommended_next_steps": [
            "Review image findings with a clinician.",
            "Compare findings with vitals, labs, and clinical notes.",
            "Use this output for research decision support only.",
        ],
        "disclaimer": "Research demo only. Not for clinical use.",
    }


def generate_summaries(
    ehr_data: pd.DataFrame,
    documents: pd.DataFrame,
    max_patients: int = 50,
) -> list[dict]:
    """Generate explainable summaries for a sample of patients."""

    summaries = []

    sample_patients = ehr_data.head(max_patients)

    for _, patient_row in sample_patients.iterrows():
        patient_summary = create_patient_summary(
            patient_row=patient_row,
            documents=documents,
        )

        summaries.append(patient_summary)

    return summaries


def save_json(data: list[dict], output_path) -> None:
    """Save summaries to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Saved: {output_path}")


def main() -> None:
    """
    Main workflow:
    1. Load EHR data
    2. Load clinical documents
    3. Build explainable patient summaries
    4. Save summaries to JSON
    """

    config = load_config()
    root = project_root()

    ehr_path = root / config["data"]["structured_ehr"]
    documents_path = root / config["data"]["clinical_documents"]

    ehr_data = pd.read_csv(ehr_path)
    documents = pd.read_csv(documents_path)

    summaries = generate_summaries(
        ehr_data=ehr_data,
        documents=documents,
        max_patients=50,
    )

    output_path = root / "outputs" / "sample_explainable_outputs.json"

    save_json(
        data=summaries,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()