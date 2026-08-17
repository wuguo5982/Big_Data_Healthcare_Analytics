# patient_agent.py
# Patient Data Agent processes structured patient, lab, and encounter data to create a summary for downstream agents. 
# It extracts relevant features such as age, sex, conditions, latest lab results, and encounter information. 
# This summary is used by other agents for risk assessment, evidence retrieval, and report generation.   

class PatientDataAgent:
    name = "patient_data_agent"

    def run(self, rows: dict) -> dict:
        p = rows["patient"]
        lab = rows["lab"]
        enc = rows["encounter"]

        summary = {
            "age": int(p.get("age", 0)),
            "sex": p.get("sex"),
            "conditions": [
                name for name, flag in {
                    "diabetes": p.get("diabetes"),
                    "heart failure": p.get("heart_failure"),
                    "COPD": p.get("copd"),
                    "CKD": p.get("ckd"),
                }.items() if int(flag or 0) == 1
            ],
            "latest_labs": {
                "creatinine_mg_dl": float(lab.get("creatinine_mg_dl", 0)),
                "hemoglobin_g_dl": float(lab.get("hemoglobin_g_dl", 0)),
                "a1c_percent": float(lab.get("a1c_percent", 0)),
            },
            "encounter": {
                "days_since_discharge": int(enc.get("days_since_discharge", 0)),
                "length_of_stay_days": int(enc.get("length_of_stay_days", 0)),
                "discharge_disposition": enc.get("discharge_disposition"),
            }
        }
        return summary
