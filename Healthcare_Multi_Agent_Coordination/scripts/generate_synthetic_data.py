from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n = 5000
patient_ids = [f"P{i:04d}" for i in range(1, n + 1)]

patients = pd.DataFrame({
    "patient_id": patient_ids,
    "age": rng.integers(35, 91, n),
    "sex": rng.choice(["F", "M"], n),
    "diabetes": rng.binomial(1, 0.34, n),
    "heart_failure": rng.binomial(1, 0.22, n),
    "copd": rng.binomial(1, 0.18, n),
    "ckd": rng.binomial(1, 0.20, n),
})

burden = patients[["diabetes","heart_failure","copd","ckd"]].sum(axis=1).to_numpy()
ed = np.clip(rng.poisson(0.8 + 0.35 * burden), 0, 10)
adm = np.clip(rng.poisson(0.35 + 0.25 * burden), 0, 8)
spec = np.clip(rng.poisson(1.5 + 0.5 * burden), 0, 12)

claims = pd.DataFrame({
    "patient_id": patient_ids,
    "ed_visits_6m": ed,
    "inpatient_admits_6m": adm,
    "specialist_visits_6m": spec,
    "allowed_amount_6m": np.round(rng.gamma(2.0, 1800, n) + 2200*adm + 650*ed, 2),
})

creatinine = np.round(rng.normal(1.0, 0.22, n) + patients["ckd"].to_numpy()*rng.uniform(0.45,1.3,n), 2)
hemoglobin = np.round(rng.normal(13.2,1.5,n) - 0.55*patients["ckd"].to_numpy(), 1)
a1c = np.round(rng.normal(5.8,0.6,n) + 1.6*patients["diabetes"].to_numpy(), 1)

labs = pd.DataFrame({
    "patient_id": patient_ids,
    "creatinine_mg_dl": np.clip(creatinine,0.5,4.5),
    "hemoglobin_g_dl": np.clip(hemoglobin,7.0,18.0),
    "a1c_percent": np.clip(a1c,4.0,13.0),
})

logit = (-4.3 + 0.025*(patients["age"].to_numpy()-50)
         + 0.65*patients["heart_failure"].to_numpy()
         + 0.42*patients["copd"].to_numpy()
         + 0.55*patients["ckd"].to_numpy()
         + 0.25*patients["diabetes"].to_numpy()
         + 0.42*ed + 0.72*adm
         + 0.55*(labs["creatinine_mg_dl"].to_numpy()>1.5))
p = 1/(1+np.exp(-logit))
labels = pd.DataFrame({"patient_id": patient_ids, "readmission_30d": rng.binomial(1,p)})

encounters = pd.DataFrame({
    "encounter_id": [f"E{i:05d}" for i in range(1,n+1)],
    "patient_id": patient_ids,
    "encounter_type": np.where(adm>0,"Inpatient","Outpatient"),
    "days_since_discharge": rng.integers(2,30,n),
    "length_of_stay_days": np.maximum(1,rng.poisson(3 + burden*0.4)),
    "discharge_disposition": rng.choice(["Home","Home Health","Skilled Nursing"], n, p=[0.70,0.20,0.10]),
})

notes = []
for i,row in patients.iterrows():
    parts = ["Recent discharge. Follow-up planning requested."]
    if row["heart_failure"]: parts.append(rng.choice(["Hx of CHF with intermittent shortness of breath.","History of congestive heart failure; mild dyspnea noted.","Prior CHF, reports SOB with exertion."]))
    if row["copd"]: parts.append(rng.choice(["COPD noted in problem list.","Chronic obstructive pulmonary disease; uses inhaler.","Possible copd exacerbation history."]))
    if row["ckd"]: parts.append(rng.choice(["CKD documented with elevated creatinine.","Chronic kidney disease in prior records.","Renal impairment / chrnic kidney dz noted."]))
    if row["diabetes"]: parts.append(rng.choice(["Type 2 diabetes mellitus; glucose monitoring discussed.","DM2 with elevated A1c.","Diabetes noted; reinforce medication reconciliation."]))
    if claims.loc[i,"ed_visits_6m"] >= 3: parts.append("Multiple recent ED visits suggest high utilization.")
    if claims.loc[i,"inpatient_admits_6m"] >= 2: parts.append("Several recent hospital admissions documented.")
    notes.append({"note_id":f"N{i+1:05d}","patient_id":row["patient_id"],"note_type":"Discharge Summary","note_text":" ".join(parts)})
clinical_notes = pd.DataFrame(notes)

topics = [
    ("post-discharge follow-up","Recent hospitalization and repeated acute-care use are important signals for care-coordination follow-up."),
    ("medication reconciliation","Medication reconciliation after transitions of care can help identify discrepancies."),
    ("heart failure","Patients with heart failure and recurrent utilization may warrant closer post-discharge monitoring."),
    ("chronic kidney disease","Kidney-function abnormalities should be interpreted with full clinical context and longitudinal laboratory history."),
    ("diabetes","Diabetes management after discharge commonly includes review of medications, monitoring plans, and follow-up arrangements."),
    ("COPD","COPD with recent acute-care utilization may indicate need for structured follow-up under local protocols."),
    ("readmission risk","Readmission-risk models should support, not replace, clinical judgment and should be monitored for calibration and drift."),
    ("AI safety","AI-generated clinical summaries should be grounded in source data and reviewed for unsupported claims."),
]
settings = ["care management","hospital discharge","payer analytics","population health","clinical operations"]

guidelines = pd.DataFrame([
    {
        "guideline_id":f"G{i+1:05d}",
        "topic":topics[i % len(topics)][0],
        "text":f"Demo evidence record {i+1}. {topics[i % len(topics)][1]} Context: {settings[i % len(settings)]}. Use approved local policy and qualified clinical judgment. Synthetic portfolio evidence only."
    }
    for i in range(n)
])

for filename, df in {
    "patients.csv":patients, "claims.csv":claims, "labs.csv":labs,
    "encounters.csv":encounters, "clinical_notes.csv":clinical_notes,
    "labels.csv":labels, "guidelines.csv":guidelines
}.items():
    df.to_csv(OUT/filename, index=False)
    print(f"{filename}: {len(df):,} rows")
