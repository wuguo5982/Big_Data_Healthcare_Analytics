from pathlib import Path
import json
import pandas as pd
import streamlit as st

from tools.insurance_prediction_tool import MODEL_PATH, predict_insurance_cost

ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Diabetes Insurance Cost Predictor",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.5rem;
        max-width: 1100px;
    }

    .title {
        color: #1f7a3f;
        font-size: 26px;
        font-weight: 700;
    }

    .green-box {
        background-color: #eaf7ee;
        border-left: 5px solid #2e8b57;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 13px;
        margin-top: 6px;
    }

    .card {
        background-color: #f6fbf7;
        border: 1px solid #cfe8d5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
    }

    div[data-testid="stMetricValue"] {
        color: #1f7a3f;
        font-size: 26px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🩺 Diabetes Insurance Cost Predictor</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="green-box">
    <b>Use Notice</b>
    <ul>
        <li>Educational and analytics use only</li>
        <li>Not a medical diagnosis, treatment recommendation, or final insurance decision</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("Patient Profile")

    age = st.slider("Age", 18, 90, 58)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", 15.0, 55.0, 32.5)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    children = st.slider("Children", 0, 5, 1)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    diabetes = st.selectbox("Diabetes", ["yes", "no"])
    hypertension = st.selectbox("Hypertension", ["yes", "no"])
    hba1c = st.slider("HbA1c", 4.0, 14.0, 8.2)

patient = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "smoker": smoker,
    "children": children,
    "region": region,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "hba1c": hba1c,
}

# Predict automatically
result = predict_insurance_cost(**patient)

left, right = st.columns([1, 1])

# ---------------- LEFT ----------------
with left:
    st.markdown("#### Input Profile")
    st.dataframe(pd.DataFrame([patient]), width="stretch", height=75)

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{metrics.get('mae', 0):,.0f}")
        m2.metric("RMSE", f"{metrics.get('rmse', 0):,.0f}")
        m3.metric("R²", f"{metrics.get('r2', 0):.3f}")
        m4.metric("Rows", f"{metrics.get('rows', 0):,}")

    st.markdown("#### Model Interpretation")
    st.markdown(
        "<div class='card'>Model estimates cost from age, BMI, smoking, diabetes, "
        "hypertension, HbA1c, region, and demographics.</div>",
        unsafe_allow_html=True,
    )

    # ✅ MOVED DISCLAIMER HERE
    st.markdown("#### Model Disclaimer")
    st.markdown(
        f"""
        <div class="green-box">
        <ul>
            <li>{result["disclaimer"]}</li>
            <li>Requires clinical, actuarial, and compliance validation</li>
            <li>Not for real-world decision-making alone</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Production Enhancements")
    st.markdown(
        "<div class='card'>SHAP | MLflow | Drift Monitoring | PHI Controls | Fairness | Human Review</div>",
        unsafe_allow_html=True,
    )

# ---------------- RIGHT ----------------
with right:
    st.markdown("#### Prediction")

    st.metric(
        "Estimated Annual Cost",
        f"${result['estimated_annual_cost_usd']:,.2f}",
    )

    st.success(f"Risk Level: {result['risk_level']}")

    st.markdown("#### Cost Drivers")
    st.markdown(
        "<div class='card'>"
        + "".join([f"<li>{d}</li>" for d in result["cost_drivers"]])
        + "</div>",
        unsafe_allow_html=True,
    )