from src.config import (
    DATA_DIR, MODEL_DIR, USE_BEDROCK, AWS_REGION, BEDROCK_MODEL_ID
)
from src.data_loader import load_data
from src.models.risk_model import ReadmissionRiskModel
from src.rag.local_retriever import LocalEvidenceRetriever
from src.llm.bedrock_client import BedrockConverseClient

from src.agents.patient_agent import PatientDataAgent
from src.agents.clinical_nlp_agent import ClinicalNLPAgent
from src.agents.claims_agent import ClaimsAgent
from src.agents.risk_agent import RiskAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.agents.safety_agent import SafetyAgent
from src.workflow import HealthcareMultiAgentWorkflow

def build_workflow():
    data = load_data(DATA_DIR)

    risk_model = ReadmissionRiskModel(
        MODEL_DIR / "readmission_random_forest.joblib"
    ).load_or_train(data)

    retriever = LocalEvidenceRetriever(data.guidelines)

    llm = None
    if USE_BEDROCK:
        llm = BedrockConverseClient(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
        )

    return HealthcareMultiAgentWorkflow(
        data=data,
        patient_agent=PatientDataAgent(),
        clinical_agent=ClinicalNLPAgent(),
        claims_agent=ClaimsAgent(),
        risk_agent=RiskAgent(risk_model),
        evidence_agent=EvidenceAgent(retriever),
        synthesis_agent=SynthesisAgent(llm_client=llm),
        safety_agent=SafetyAgent(),
    )
