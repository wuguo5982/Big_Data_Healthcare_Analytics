import pandas as pd

FEATURES = [
    "age", "diabetes", "heart_failure", "copd", "ckd",
    "ed_visits_6m", "inpatient_admits_6m", "specialist_visits_6m",
    "allowed_amount_6m", "creatinine_mg_dl", "hemoglobin_g_dl", "a1c_percent"
]

def build_training_frame(data) -> pd.DataFrame:
    df = (
        data.patients
        .merge(data.claims, on="patient_id")
        .merge(data.labs, on="patient_id")
        .merge(data.labels, on="patient_id")
    )
    return df

def patient_feature_frame(patient: dict, claim: dict, lab: dict) -> pd.DataFrame:
    row = {**patient, **claim, **lab}
    return pd.DataFrame([{k: row.get(k, 0) for k in FEATURES}])
