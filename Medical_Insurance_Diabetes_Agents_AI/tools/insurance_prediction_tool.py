from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "insurance_diabetes_linear_model.json"

VALID_VALUES = {
    "sex": {"male", "female"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
    "diabetes": {"yes", "no"},
    "hypertension": {"yes", "no"},
}


def _normalize_yes_no(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return "yes"
    if value in {"n", "no", "false", "0"}:
        return "no"
    return value


def validate_patient_input(patient: Dict[str, Any]) -> Dict[str, Any]:
    required = ["age", "sex", "bmi", "smoker", "children", "region", "diabetes", "hypertension", "hba1c"]
    missing = [k for k in required if k not in patient or patient[k] in [None, ""]]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    clean = {
        "age": int(patient["age"]),
        "sex": str(patient["sex"]).strip().lower(),
        "bmi": float(patient["bmi"]),
        "smoker": _normalize_yes_no(patient["smoker"]),
        "children": int(patient["children"]),
        "region": str(patient["region"]).strip().lower(),
        "diabetes": _normalize_yes_no(patient["diabetes"]),
        "hypertension": _normalize_yes_no(patient["hypertension"]),
        "hba1c": float(patient["hba1c"]),
    }

    if not (18 <= clean["age"] <= 100):
        raise ValueError("age must be between 18 and 100")
    if not (10 <= clean["bmi"] <= 70):
        raise ValueError("bmi must be between 10 and 70")
    if not (0 <= clean["children"] <= 20):
        raise ValueError("children must be between 0 and 20")
    if not (3.5 <= clean["hba1c"] <= 15):
        raise ValueError("hba1c must be between 3.5 and 15")

    for field, valid_set in VALID_VALUES.items():
        if clean[field] not in valid_set:
            raise ValueError(f"{field} must be one of {sorted(valid_set)}")

    return clean


def _load_model() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python training/train_model.py")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _build_design_row(patient: Dict[str, Any], model: Dict[str, Any]) -> np.ndarray:
    values = []
    for name in model["feature_names"]:
        if name == "intercept":
            values.append(1.0)
        elif name in patient:
            values.append(float(patient[name]))
        elif "__" in name:
            col, level = name.split("__", 1)
            values.append(1.0 if str(patient[col]).lower() == level else 0.0)
        else:
            raise ValueError(f"Unknown model feature: {name}")
    return np.array(values, dtype=float)


def infer_risk_level(cost: float) -> str:
    if cost < 10000:
        return "Low"
    if cost < 20000:
        return "Moderate"
    if cost < 30000:
        return "High"
    return "Very High"


def identify_cost_drivers(patient: Dict[str, Any]) -> List[str]:
    drivers: List[str] = []
    if patient["age"] >= 55:
        drivers.append("Older age may increase expected healthcare utilization.")
    if patient["bmi"] >= 30:
        drivers.append("BMI >= 30 is associated with obesity-related cost risk.")
    if patient["smoker"] == "yes":
        drivers.append("Smoking is associated with higher long-term medical costs.")
    if patient["diabetes"] == "yes":
        drivers.append("Diabetes can increase monitoring, medication, and complication-related costs.")
    if patient["hypertension"] == "yes":
        drivers.append("Hypertension can add cardiovascular risk and related care costs.")
    if patient["hba1c"] >= 7.0:
        drivers.append("Elevated HbA1c may indicate higher diabetes-related risk.")
    return drivers or ["No major high-risk cost driver was detected from the provided fields."]


def predict_insurance_cost(
    age: int,
    sex: str,
    bmi: float,
    smoker: str,
    children: int,
    region: str,
    diabetes: str,
    hypertension: str,
    hba1c: float,
) -> Dict[str, Any]:
    """Predict annual medical insurance cost from patient features."""
    patient = validate_patient_input(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "smoker": smoker,
            "children": children,
            "region": region,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "hba1c": hba1c,
        }
    )

    model = _load_model()
    x = _build_design_row(patient, model)
    beta = np.array(model["coefficients"], dtype=float)
    estimated_cost = max(float(x @ beta), 0.0)

    return {
        "estimated_annual_cost_usd": round(estimated_cost, 2),
        "risk_level": infer_risk_level(estimated_cost),
        "cost_drivers": identify_cost_drivers(patient),
        "input": patient,
        "disclaimer": (
            "This is an ML-based estimate for education/analytics only. "
            "It is not a medical diagnosis, treatment recommendation, or final insurance pricing decision."
        ),
    }
