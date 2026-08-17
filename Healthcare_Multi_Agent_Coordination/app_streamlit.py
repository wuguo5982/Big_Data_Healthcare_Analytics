import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
)

st.title("Multi-Agent Clinical Risk & Care Coordination System")
st.caption(
    "Synthetic-data portfolio demo — decision support only; "
    "not for diagnosis or prescribing."
)

try:
    from src.factory import build_workflow
except Exception as exc:
    st.error("The application could not import the workflow.")
    st.exception(exc)
    st.info(
        "Run `python check_environment.py`, then reinstall with "
        "`python -m pip install -r requirements.txt`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading data and risk model...")
def get_workflow():
    return build_workflow()


try:
    workflow = get_workflow()
except Exception as exc:
    st.error("The workflow failed during initialization.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Patient Selection")
    patient_id = st.text_input("Patient ID", "P0095")
    st.caption("Synthetic patient IDs range from P0001 to P5000.")

query = st.text_area(
    "Care-manager question",
    "Summarize readmission risk and care-coordination priorities.",
    height=100,
)

if st.button("Run multi-agent workflow", type="primary"):
    try:
        with st.spinner("Agents are analyzing the patient record..."):
            result = workflow.invoke(patient_id.strip(), query.strip())

        risk = result["risk_summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "30-Day Readmission Risk",
            f"{risk['readmission_probability']:.1%}",
        )
        c2.metric("Risk Level", risk["risk_level"])
        auc = risk.get("validation_auc")
        c3.metric("Validation AUROC", f"{auc:.3f}" if auc is not None else "N/A")

        st.subheader("Final Care-Coordination Summary")
        st.info(result["final_report"])

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Patient", "Clinical NLP", "Claims", "Evidence / RAG", "Safety"]
        )
        with tab1:
            st.json(result["patient_summary"])
        with tab2:
            st.json(result["clinical_summary"])
        with tab3:
            st.json(result["claims_summary"])
        with tab4:
            st.json(result["evidence_summary"])
        with tab5:
            if result["safety"]["passed"]:
                st.success("Safety validation passed.")
            else:
                st.warning("Safety validation required repair.")
            st.json(result["safety"])

    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("The multi-agent workflow encountered an unexpected error.")
        st.exception(exc)

st.divider()
st.caption("All patient records and evidence in this repository are synthetic and PHI-free.")
