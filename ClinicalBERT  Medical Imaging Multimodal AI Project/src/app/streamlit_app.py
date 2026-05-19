from pathlib import Path
import sys

from PIL import Image
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from utils import (
    load_config,
    project_root,
    load_ehr,
    load_documents,
    build_patient_summary,
)


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make dataframe safe for Streamlit display."""

    df = df.copy()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str)

    return df


st.set_page_config(
    page_title="NIH ClinicalBERT Demo",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }

    div[data-baseweb="select"] > div {
        font-size: 20px !important;
    }

    table {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NIH ChestX-ray14 + ClinicalBERT Multimodal AI")
st.caption("Research demo only. Not for clinical use.")

cfg = load_config()
ehr = load_ehr()
docs = load_documents()

st.sidebar.header("Select Patient")

patient_id = st.sidebar.selectbox(
    "Patient ID",
    ehr["patient_id"].tolist(),
)

patient_summary = build_patient_summary(patient_id)
patient_row = ehr[ehr["patient_id"] == patient_id].iloc[0]

left_col, right_col = st.columns([1.0, 1.3])

with left_col:
    st.subheader("Explainable Summary")
    st.json(patient_summary)

    st.subheader("Structured EHR")

    ehr_display = patient_row.to_frame(name="value").reset_index()
    ehr_display.columns = ["feature", "value"]
    ehr_display = make_arrow_safe(ehr_display)

    st.dataframe(
        ehr_display,
        width="stretch",
        height=450,
    )

with right_col:
    st.subheader("NIH Chest X-ray")

    image_path = (
        project_root()
        / cfg["nih"]["selected_images_dir"]
        / str(patient_row["image_id"])
    )

    if image_path.exists():
        st.image(
            Image.open(image_path),
            caption=str(patient_row["image_id"]),
            width=700,
        )
    else:
        st.warning("Image missing. Run data preparation first.")

    st.subheader("Clinical Documents")

    patient_docs = docs.loc[
        docs["patient_id"] == patient_id,
        ["note_type", "text"],
    ]

    patient_docs = make_arrow_safe(patient_docs)

    st.dataframe(
        patient_docs,
        width="stretch",
        height=350,
    )

st.subheader("Risk Distribution")

risk_counts = (
    ehr["high_risk_label"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

risk_chart = pd.DataFrame(
    {
        "Risk Group": ["Low", "High"],
        "Count": risk_counts.values,
    }
)

st.bar_chart(
    risk_chart.set_index("Risk Group"),
    height=250,
)