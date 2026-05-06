"""
Lightweight data validation for CI/CD.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "data" / "raw" / "claims.csv"

REQUIRED_COLUMNS = {
    "claim_id",
    "provider_id",
    "procedure_code",
    "diagnosis_code",
    "amount_billed",
    "amount_paid",
    "risk_score",
    "risk_label",
}

def main() -> None:
    if not CLAIMS_PATH.exists():
        print(f"Warning: {CLAIMS_PATH} not found. Skipping strict claims validation.")
        return

    df = pd.read_csv(CLAIMS_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"claims.csv missing required columns: {sorted(missing)}")

    if len(df) < 50:
        raise ValueError("claims.csv should contain at least 50 rows for demo validation.")

    if "risk_score" in df.columns and not df["risk_score"].between(0, 1).all():
        raise ValueError("risk_score must be between 0 and 1.")

    print("Data validation passed.")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
