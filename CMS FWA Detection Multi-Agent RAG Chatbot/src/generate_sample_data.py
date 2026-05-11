"""Generate synthetic CMS-style claims, provider, beneficiary, and doctor-note data.
This project uses synthetic/sample data only. Replace with CMS SynPUF, data.cms.gov,
data.medicaid.gov, Synthea, or approved internal claims data for production.
"""
from __future__ import annotations
from pathlib import Path
import random
import pandas as pd
import numpy as np

RNG = np.random.default_rng(42)
random.seed(42)

SPECIALTIES = ["Cardiology", "Oncology", "Primary Care", "DME Supplier", "Home Health", "Hospice", "Lab", "Pain Management"]
CPT = ["99213", "99214", "99215", "93000", "80053", "97110", "E0260", "E1390", "G0151", "Q5001", "A0428"]
ICD = ["E11.9", "I10", "C50.9", "J44.9", "M54.5", "R26.2", "Z51.5", "N18.3", "G89.4"]


def generate(base_dir: str = "data/raw", n_claims: int = 1800) -> None:
    out = Path(base_dir)
    out.mkdir(parents=True, exist_ok=True)

    providers = []
    for i in range(60):
        specialty = random.choice(SPECIALTIES)
        providers.append({
            "provider_id": f"P{i+1:04d}",
            "npi": f"1999{i+100000:06d}",
            "provider_name": f"Provider Group {i+1}",
            "specialty": specialty,
            "state": random.choice(["GA", "FL", "TX", "NY", "CA", "OH", "NC"]),
            "risk_region": random.choice(["low", "medium", "high"]),
        })
    pd.DataFrame(providers).to_csv(out / "providers.csv", index=False)

    beneficiaries = []
    for i in range(500):
        age = int(RNG.integers(21, 94))
        beneficiaries.append({
            "beneficiary_id": f"B{i+1:05d}",
            "age": age,
            "sex": random.choice(["F", "M"]),
            "dual_eligible": random.choice([0, 0, 0, 1]),
            "chronic_condition_count": int(RNG.integers(0, 8)),
            "state": random.choice(["GA", "FL", "TX", "NY", "CA", "OH", "NC"]),
        })
    pd.DataFrame(beneficiaries).to_csv(out / "beneficiaries.csv", index=False)

    suspicious_providers = {"P0004", "P0012", "P0025", "P0041"}
    rows = []
    for i in range(n_claims):
        provider = random.choice(providers)
        bene = random.choice(beneficiaries)
        service = random.choice(CPT)
        diagnosis = random.choice(ICD)
        base_charge = {
            "99213": 95, "99214": 150, "99215": 260, "93000": 70, "80053": 45,
            "97110": 85, "E0260": 850, "E1390": 450, "G0151": 170, "Q5001": 240, "A0428": 600,
        }[service]
        charge = float(max(20, RNG.normal(base_charge, base_charge * 0.25)))
        paid = float(charge * RNG.uniform(0.45, 0.9))
        units = int(max(1, RNG.poisson(1.3)))
        days_ago = int(RNG.integers(1, 360))
        fwa_label = 0
        flags = []

        if provider["provider_id"] in suspicious_providers:
            # Inject realistic suspicious behavior: excessive DME, upcoding, duplicates, high charges
            if RNG.random() < 0.55:
                service = random.choice(["E0260", "E1390", "99215", "A0428"])
                charge *= RNG.uniform(2.0, 4.2)
                paid *= RNG.uniform(1.7, 3.0)
                units = int(max(units, RNG.integers(2, 6)))
                fwa_label = 1
                flags.append("excessive_high_cost_or_dme")
        if service == "99215" and diagnosis in ["E11.9", "I10"] and RNG.random() < 0.40:
            fwa_label = 1
            flags.append("possible_upcoding")
        if service in ["E0260", "E1390"] and diagnosis not in ["J44.9", "R26.2", "Z51.5"] and RNG.random() < 0.45:
            fwa_label = 1
            flags.append("medical_necessity_mismatch")

        rows.append({
            "claim_id": f"C{i+1:07d}",
            "provider_id": provider["provider_id"],
            "beneficiary_id": bene["beneficiary_id"],
            "claim_type": random.choice(["Medicare", "Medicaid", "Dual"]),
            "service_date": (pd.Timestamp.today().normalize() - pd.Timedelta(days=days_ago)).date().isoformat(),
            "cpt_hcpcs_code": service,
            "icd10_code": diagnosis,
            "units": units,
            "submitted_charge": round(charge * units, 2),
            "allowed_amount": round(paid, 2),
            "paid_amount": round(paid * RNG.uniform(0.75, 1.0), 2),
            "place_of_service": random.choice(["office", "hospital", "home", "telehealth", "lab"]),
            "is_fwa_synthetic_label": fwa_label,
            "injected_reason": ";".join(flags),
        })

    # duplicate claims injection
    for j in range(35):
        src = random.choice(rows).copy()
        src["claim_id"] = f"CDUP{j+1:05d}"
        src["is_fwa_synthetic_label"] = 1
        src["injected_reason"] = (src.get("injected_reason") + ";" if src.get("injected_reason") else "") + "duplicate_claim"
        rows.append(src)

    claims = pd.DataFrame(rows)
    claims.to_csv(out / "claims.csv", index=False)

    note_rows = []
    sample_size = min(n_claims, len(claims))
    # for _, r in claims.sample(420, random_state=42).iterrows():
    for _, r in claims.sample(sample_size, random_state=42, replace=False).iterrows():
        supports = not (r["cpt_hcpcs_code"] in ["E0260", "E1390"] and r["icd10_code"] not in ["J44.9", "R26.2", "Z51.5"])
        if supports:
            note = f"Patient evaluated for diagnosis {r['icd10_code']}. Documentation supports service {r['cpt_hcpcs_code']} based on symptoms, assessment, and care plan."
        else:
            note = f"Patient seen for routine follow-up. Ambulates independently. No oxygen dependence or documented mobility limitation. Billed code {r['cpt_hcpcs_code']} may require additional medical necessity documentation."
        note_rows.append({
            "note_id": f"N{len(note_rows)+1:06d}",
            "claim_id": r["claim_id"],
            "beneficiary_id": r["beneficiary_id"],
            "provider_id": r["provider_id"],
            "clinical_note": note,
        })
    pd.DataFrame(note_rows).to_csv(out / "doctor_notes.csv", index=False)

    reference = pd.DataFrame([
        {"code": "99213", "description": "Established patient office visit, low/moderate complexity"},
        {"code": "99214", "description": "Established patient office visit, moderate complexity"},
        {"code": "99215", "description": "Established patient office visit, high complexity"},
        {"code": "E0260", "description": "Hospital bed, semi-electric, durable medical equipment"},
        {"code": "E1390", "description": "Oxygen concentrator, durable medical equipment"},
        {"code": "Q5001", "description": "Hospice or home health service location"},
        {"code": "A0428", "description": "Ambulance service, basic life support, non-emergency"},
    ])
    reference.to_csv(out / "code_reference.csv", index=False)
    print(f"Generated sample data in {out.resolve()}")

if __name__ == "__main__":
    generate()
