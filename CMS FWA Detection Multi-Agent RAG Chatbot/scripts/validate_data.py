"""
Lightweight data validation for CI/CD.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

REQUIRED_FILES = {
    "claims.csv": {"claim_id", "provider_id"},
    "providers.csv": {"provider_id"},
    "beneficiaries.csv": {"beneficiary_id"},
    "doctor_notes.csv": {"claim_id"},
    "code_reference.csv": set(),
}

def main() -> None:
    missing_files = []
    for filename, required_cols in REQUIRED_FILES.items():
        path = RAW / filename
        if not path.exists():
            missing_files.append(filename)
            continue

        df = pd.read_csv(path)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"{filename} missing columns: {sorted(missing_cols)}")

        if len(df) == 0:
            raise ValueError(f"{filename} is empty")

        print(f"Validated {filename}: {len(df)} rows")

    if missing_files:
        raise FileNotFoundError(f"Missing required raw data files: {missing_files}")

    print("Data validation passed.")

if __name__ == "__main__":
    main()
