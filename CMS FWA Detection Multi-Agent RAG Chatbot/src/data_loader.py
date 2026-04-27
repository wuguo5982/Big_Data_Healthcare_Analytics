from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import RAW_DIR

def load_claims(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "claims.csv", parse_dates=["service_date"])

def load_providers(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "providers.csv")

def load_beneficiaries(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "beneficiaries.csv")

def load_notes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "doctor_notes.csv")

def load_reference(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "code_reference.csv")

def load_all(raw_dir: Path = RAW_DIR):
    return {
        "claims": load_claims(raw_dir),
        "providers": load_providers(raw_dir),
        "beneficiaries": load_beneficiaries(raw_dir),
        "notes": load_notes(raw_dir),
        "reference": load_reference(raw_dir),
    }
