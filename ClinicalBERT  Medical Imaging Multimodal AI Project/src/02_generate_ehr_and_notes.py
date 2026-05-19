"""Generate structured EHR + notes from NIH labels ("NIH labels" are the disease/finding labels from the NIH ChestX-ray14 metadata CSV)."""
from datetime import datetime, timedelta
import random, numpy as np, pandas as pd
from utils import load_config, project_root, save_json

rng = np.random.default_rng(42); random.seed(42)

def primary(row):
    for col, name in [("pneumonia","Pneumonia"),("edema","Pulmonary Edema"),("cardiomegaly","Cardiomegaly"),("effusion","Pleural Effusion"),("consolidation","Consolidation"),("atelectasis","Atelectasis"),("mass","Mass"),("nodule","Nodule"),("pneumothorax","Pneumothorax")]:
        if col in row and row[col] == 1: return name
    return "No Acute Finding"

def findings(row):
    m = {"cardiomegaly":"enlarged cardiac silhouette","edema":"interstitial prominence suggesting edema","pneumonia":"opacity concerning for pneumonia","effusion":"small pleural effusion","atelectasis":"linear basilar atelectasis","consolidation":"patchy consolidation","pneumothorax":"pleural line suspicious for pneumothorax","mass":"mass-like opacity","nodule":"nodular opacity"}
    out = [v for k,v in m.items() if k in row and row[k] == 1]
    return "; ".join(out) if out else "no acute cardiopulmonary abnormality"

def main():
    cfg, root = load_config(), project_root()
    labels = pd.read_csv(root / cfg["data"]["image_labels"])
    ehr_rows, doc_rows = [], []
    start = datetime(2024,1,1)
    for idx, row in labels.iterrows():
        age = int(row.get("patient_age", 60))
        if age <= 0 or age > 100: age = int(np.clip(rng.normal(62,15),18,90))
        sex = row.get("patient_gender", "Unknown")
        pneumonia, edema, cardiomegaly = int(row.get("pneumonia",0)), int(row.get("edema",0)), int(row.get("cardiomegaly",0))
        effusion, consolidation, atelectasis = int(row.get("effusion",0)), int(row.get("consolidation",0)), int(row.get("atelectasis",0))
        spo2 = int(np.clip(rng.normal(95 - 3*pneumonia - 2*edema, 3),75,100))
        hr = int(np.clip(rng.normal(82 + 8*pneumonia + 5*edema,14),45,160))
        rr = int(np.clip(rng.normal(18 + 4*pneumonia + 2*edema,4),10,38))
        sbp = int(np.clip(rng.normal(128 + .25*(age-50),16),85,210))
        dbp = int(np.clip(rng.normal(78,10),45,125))
        temp = round(float(np.clip(rng.normal(98.5 + 1.2*pneumonia,1.1),95,104)),1)
        wbc = round(float(np.clip(rng.normal(8.5 + 4*pneumonia,2.6),2,28)),1)
        bnp = int(np.clip(rng.lognormal(5.1 + .7*edema + .5*cardiomegaly,.9),5,4000))
        lvef = int(np.clip(rng.normal(57 - 9*cardiomegaly - 5*edema,10),15,75))
        creat = round(float(np.clip(rng.lognormal(.05,.35),.4,4.8)),2)
        glucose = int(np.clip(rng.normal(116,35),55,420))
        trop = round(float(np.clip(rng.lognormal(-3.3+.3*cardiomegaly,.95),.001,5)),3)
        qrs = int(np.clip(rng.normal(96,20),60,190)); qt = int(np.clip(rng.normal(420,32),320,560))
        comorb = int(row["abnormal_cxr_label"] + (glucose>155) + (sbp>145) + (creat>1.5))
        risk = .018*age + .8*pneumonia + .7*edema + .65*cardiomegaly + .5*effusion + .4*consolidation + .35*atelectasis + .45*(spo2<92) + .4*(lvef<45) + .25*(creat>1.5) + .015*max(0,bnp-300)/50 + rng.normal(0,.12)
        pc = primary(row)
        ehr_rows.append({"patient_id":row.patient_id,"encounter_id":row.encounter_id,"image_id":row.image_id,"age":age,"sex":sex,"bmi":round(float(np.clip(rng.normal(29,6),17,52)),1),"systolic_bp":sbp,"diastolic_bp":dbp,"heart_rate":hr,"spo2":spo2,"resp_rate":rr,"temperature_f":temp,"creatinine":creat,"glucose":glucose,"bnp":bnp,"troponin":trop,"wbc":wbc,"lvef":lvef,"qrs_ms":qrs,"qt_ms":qt,"abnormal_cxr_label":int(row.abnormal_cxr_label),"comorbidity_count":comorb,"primary_condition":pc,"risk_score":round(float(risk),3)})
        texts = [
            ("clinical_note", f"Patient {row.patient_id} is a {age}-year-old {sex}. Vitals BP {sbp}/{dbp}, HR {hr}, SpO2 {spo2}, RR {rr}. Labs BNP {bnp}, troponin {trop}, WBC {wbc}, creatinine {creat}. Concern for {pc}."),
            ("radiology_report", f"Chest radiograph report. Findings: {findings(row)}. Impression: {pc}."),
            ("discharge_summary", f"Discharge summary: primary phenotype {pc}. Risk score {risk:.2f}. Follow-up should consider imaging, vitals, and labs.")
        ]
        for j,(typ,text) in enumerate(texts):
            doc_rows.append({"document_id":f"DOC_{idx+1:05d}_{j+1}","patient_id":row.patient_id,"encounter_id":row.encounter_id,"image_id":row.image_id,"note_type":typ,"document_date":(start+timedelta(days=int(idx%365))).strftime("%Y-%m-%d"),"author_team":random.choice(["Radiology","Pulmonology","Cardiology","Emergency"]),"primary_condition":pc,"text":text})
    ehr = pd.DataFrame(ehr_rows)
    ehr["high_risk_label"] = (ehr["risk_score"] >= ehr["risk_score"].quantile(.70)).astype(int)
    docs = pd.DataFrame(doc_rows).merge(ehr[["patient_id","high_risk_label"]], on="patient_id", how="left")
    ehr.to_csv(root / cfg["data"]["structured_ehr"], index=False)
    docs.to_csv(root / cfg["data"]["clinical_documents"], index=False)
    save_json({"project_title":"NIH ChestX-ray14 + ClinicalBERT","ehr_rows":len(ehr),"documents":len(docs),"images":len(labels),"target":"high_risk_label"}, root / cfg["data"]["annotations"])
    print(f"Generated {len(ehr)} EHR rows and {len(docs)} documents.")

if __name__ == "__main__":
    main()
