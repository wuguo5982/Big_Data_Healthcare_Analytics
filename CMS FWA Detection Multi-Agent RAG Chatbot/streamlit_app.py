from __future__ import annotations
import streamlit as st
from src.config import RAW_DIR
from src.generate_sample_data import generate
from src.data_loader import load_all
from agents.fwa_agents import CMSFWAOrchestrator
from src.fraud_models import score_providers
from src.graph_detection import graph_risk_features
from src.feature_engineering import add_claim_flags

st.set_page_config(page_title="CMS FWA Multi-Agent RAG Chatbot", layout="wide")
st.title("CMS Fraud, Waste & Abuse Detection — Multi-Agent RAG Chatbot")
st.caption("Synthetic demo using CMS-style claims, doctor notes, policy PDFs, anomaly detection, and grounded RAG citations.")

if not (RAW_DIR / "claims.csv").exists():
    generate(str(RAW_DIR))

data = load_all()
claims, providers, notes = data["claims"], data["providers"], data["notes"]
orch = CMSFWAOrchestrator(claims, providers, notes)

with st.sidebar:
    st.header("Project Controls")
    if st.button("Regenerate synthetic CMS-style data"):
        generate(str(RAW_DIR))
        st.success("Data regenerated. Refresh the app.")
    st.markdown("### Example questions")
    st.code("""Which providers are most suspicious?
Review provider P0004
Review claim CDUP00001
What is fraud vs abuse?
Which DME patterns look risky?""")

scores = score_providers(claims, providers)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Claims", f"{len(claims):,}")
col2.metric("Providers", f"{providers['provider_id'].nunique():,}")
col3.metric("Doctor Notes", f"{len(notes):,}")
col4.metric("High-Risk Providers", f"{(scores['risk_level'].astype(str) == 'High').sum():,}")

tab1, tab2, tab3, tab4 = st.tabs(["Chatbot", "Provider Risk", "Claim Explorer", "Architecture"])

with tab1:
    q = st.text_area("Ask the CMS FWA assistant", value="Which providers are most suspicious for potential FWA review?", height=90)
    if st.button("Run Multi-Agent Analysis", type="primary"):
        resp = orch.answer(q)
        st.subheader("Answer")
        st.text(resp.answer)
        st.info(f"Grounding: {resp.grounding_status} | Confidence: {resp.confidence:.2f} | Citations: {resp.citations}")

with tab2:
    st.subheader("Provider-level anomaly and FWA risk ranking")
    st.dataframe(scores.head(25), use_container_width=True)
    st.bar_chart(scores.set_index("provider_id")["risk_score"].head(15))
    st.subheader("Provider-beneficiary graph risk")
    st.dataframe(graph_risk_features(claims).head(20), use_container_width=True)

with tab3:
    st.subheader("Claims with rule flags")
    display = add_claim_flags(claims.copy())
    flagged = display[(display["duplicate_claim_flag"] == 1) | (display["dme_dx_mismatch_flag"] == 1) | (display["high_complexity_low_dx_flag"] == 1) | (display["high_paid_flag"] == 1)]
    st.dataframe(flagged.head(100), use_container_width=True)

with tab4:
    st.markdown("""
```text
CMS / Medicaid / SynPUF / Synthea / Kaggle-style data
        ↓
Data Ingestion + Cleaning + Feature Engineering
        ↓
Structured Claims Analytics + Doctor Notes NLP
        ↓
Policy PDF RAG Retrieval with Citations
        ↓
ML Anomaly Detection + Rule Engine + Graph Risk
        ↓
Multi-Agent Orchestration
        ↓
Grounded Chatbot + Investigator Report + Human Review
```
""")
