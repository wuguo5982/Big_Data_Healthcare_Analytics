from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataBundle:
    patients: pd.DataFrame
    claims: pd.DataFrame
    labs: pd.DataFrame
    encounters: pd.DataFrame
    clinical_notes: pd.DataFrame
    labels: pd.DataFrame
    guidelines: pd.DataFrame

def load_data(data_dir: Path) -> DataBundle:
    return DataBundle(
        patients=pd.read_csv(data_dir / "patients.csv"),
        claims=pd.read_csv(data_dir / "claims.csv"),
        labs=pd.read_csv(data_dir / "labs.csv"),
        encounters=pd.read_csv(data_dir / "encounters.csv"),
        clinical_notes=pd.read_csv(data_dir / "clinical_notes.csv"),
        labels=pd.read_csv(data_dir / "labels.csv"),
        guidelines=pd.read_csv(data_dir / "guidelines.csv"),
    )

def get_patient_rows(data: DataBundle, patient_id: str) -> dict:
    def first(df):
        rows = df[df["patient_id"] == patient_id]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    notes = data.clinical_notes[data.clinical_notes["patient_id"] == patient_id]
    return {
        "patient": first(data.patients),
        "claim": first(data.claims),
        "lab": first(data.labs),
        "encounter": first(data.encounters),
        "notes": notes.to_dict(orient="records"),
    }
