# synthesis_agent.py
# Synthesis Agent generates a comprehensive report summarizing patient data, clinical findings, risk assessment, and supporting evidence. 
# It can operate in a local deterministic mode or use an LLM for more nuanced report generation, 
# ensuring that the output is decision support only and not a diagnosis or prescription. 

import json

class SynthesisAgent:
    name = "synthesis_agent"

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _local_report(self, state: dict) -> str:
        p = state["patient_summary"]
        c = state["clinical_summary"]
        u = state["claims_summary"]
        r = state["risk_summary"]
        evidence = state["evidence_summary"]["evidence"]

        conditions = ", ".join(c["detected_conditions"]) or "none detected in demo note"
        citations = ", ".join(e["guideline_id"] for e in evidence)

        return (
            f"Patient {state['patient_id']} has a {r['risk_level']} estimated 30-day "
            f"readmission risk ({r['readmission_probability']:.1%}). "
            f"Detected clinical conditions: {conditions}. "
            f"Recent utilization includes {u['ed_visits_6m']} ED visit(s) and "
            f"{u['inpatient_admits_6m']} inpatient admission(s) in 6 months. "
            f"Latest creatinine is {p['latest_labs']['creatinine_mg_dl']} mg/dL. "
            f"Care-coordination review should focus on recent utilization, transition-of-care "
            f"needs, and the organization's approved follow-up pathway. "
            f"Evidence references: {citations}. "
            f"This output is decision support only and is not a diagnosis or prescription."
        )

    def run(self, state: dict) -> dict:
        if self.llm_client is None:
            return {
                "report": self._local_report(state),
                "generation_mode": "deterministic local demo",
            }

        system_prompt = (
            "You are a healthcare decision-support summarizer. "
            "Do not diagnose, prescribe, or invent facts. Use only supplied patient data "
            "and evidence. Clearly label uncertainty and include evidence IDs."
        )
        payload = {
            "question": state["query"],
            "patient": state["patient_summary"],
            "clinical_nlp": state["clinical_summary"],
            "claims": state["claims_summary"],
            "risk": state["risk_summary"],
            "evidence": state["evidence_summary"]["evidence"],
        }
        report = self.llm_client.generate(system_prompt, json.dumps(payload, indent=2))
        return {"report": report, "generation_mode": "Amazon Bedrock Converse"}
