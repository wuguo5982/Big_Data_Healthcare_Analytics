# evidence_agent.py
# Evidence Agent retrieves relevant clinical guidelines and evidence based on patient data, clinical notes, and risk assessment. 
# It uses a retriever to query a knowledge base for supporting evidence. 

class EvidenceAgent:
    name = "evidence_agent"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, patient_summary: dict, clinical_summary: dict, risk_summary: dict):
        conditions = ", ".join(clinical_summary.get("detected_conditions", []))
        search_query = (
            f"{query}. Patient conditions: {conditions}. "
            f"Risk: {risk_summary.get('risk_level')}. "
            f"Discharge and readmission care coordination."
        )
        return {
            "query": search_query,
            "evidence": self.retriever.retrieve(search_query, k=3),
        }
