
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.enterprise.production_workflow import run_production_analysis
from src.llm.gpt4omini_engine import is_openai_ready
from src.evaluation.document_analytics import (
    load_claims,
    load_clinical_notes,
    load_policy_text,
    load_audit_rules,
    provider_risk_summary,
)


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Improved Healthcare FWA with Grounded RAG",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid #e6e8eb;
        border-radius: 14px;
        padding: 18px 18px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        min-height: 112px;
    }
    .metric-label {font-size: 0.82rem; color: #60646c; margin-bottom: 0.4rem;}
    .metric-value {font-size: 1.55rem; font-weight: 750; color: #1f2937;}
    .metric-help {font-size: 0.78rem; color: #7a7f87; margin-top: 0.35rem;}
    .section-card {
        border: 1px solid #e6e8eb;
        border-radius: 14px;
        padding: 18px;
        background: #fbfcfd;
        margin-top: 0.5rem;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        background: #eef6ff;
        color: #0f4c81;
        font-size: 0.82rem;
        font-weight: 650;
        margin-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def fmt_money(value: float) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def clean_risk_label(value) -> str:
    value = str(value).strip().lower()
    if value in {"high", "h", "1"}:
        return "High"
    if value in {"medium", "med", "m"}:
        return "Medium"
    if value in {"low", "l", "0"}:
        return "Low"
    return value.title() if value else "Unknown"


def prepare_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "provider_id" not in df.columns:
        df["provider_id"] = "P_UNKNOWN"

    if "claim_id" not in df.columns:
        df["claim_id"] = [f"C{i:05d}" for i in range(1, len(df) + 1)]

    if "amount_billed" not in df.columns:
        df["amount_billed"] = 0.0

    if "amount_paid" not in df.columns:
        df["amount_paid"] = df["amount_billed"] * 0.45

    if "risk_score" not in df.columns:
        max_billed = df["amount_billed"].max() if len(df) else 1
        df["risk_score"] = (df["amount_billed"] / max_billed).clip(0, 1) if max_billed else 0.3

    if "risk_label" not in df.columns:
        df["risk_label"] = pd.cut(
            df["risk_score"],
            bins=[-0.01, 0.45, 0.75, 1.01],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    else:
        df["risk_label"] = df["risk_label"].apply(clean_risk_label)

    if "procedure_code" not in df.columns:
        df["procedure_code"] = "UNKNOWN"

    if "diagnosis_code" not in df.columns:
        df["diagnosis_code"] = "UNKNOWN"

    df["amount_billed"] = pd.to_numeric(df["amount_billed"], errors="coerce").fillna(0)
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0).clip(0, 1)
    df["provider_id"] = df["provider_id"].astype(str)

    return df


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame, providers, risk_labels, min_risk, amount_range):
    out = df.copy()
    if providers:
        out = out[out["provider_id"].isin(providers)]
    if risk_labels:
        out = out[out["risk_label"].isin(risk_labels)]
    out = out[out["risk_score"] >= min_risk]
    out = out[
        (out["amount_billed"] >= amount_range[0])
        & (out["amount_billed"] <= amount_range[1])
    ]
    return out


def top_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "claim_id",
        "provider_id",
        "procedure_code",
        "diagnosis_code",
        "amount_billed",
        "amount_paid",
        "risk_label",
        "risk_score",
    ]
    return [c for c in cols if c in df.columns]


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
claims_df = prepare_claims(load_claims())
notes_df = load_clinical_notes()
policy_text = load_policy_text()
audit_rules = load_audit_rules()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Improved Healthcare FWA with Grounded RAG")
st.caption(
    "A clean production-style dashboard for claims analytics, document intelligence, provider drill-down, "
    "and grounded GPT-4o-mini FWA reasoning."
)

st.markdown(
    """
    <span class="status-pill">Claims Analytics</span>
    <span class="status-pill">Document Analytics</span>
    <span class="status-pill">Provider Drill-Down</span>
    <span class="status-pill">RAG + GPT-4o-mini</span>
    <span class="status-pill">Hallucination Reduction</span>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    if is_openai_ready():
        st.success("GPT-4o-mini API ready")
    else:
        st.warning("GPT-4o-mini API not configured")
        st.caption("The app still runs with a safe fallback. Set OPENAI_API_KEY for live reasoning.")

    st.divider()

    provider_options = sorted(claims_df["provider_id"].unique().tolist())
    risk_options = ["High", "Medium", "Low"]
    risk_options = [r for r in risk_options if r in claims_df["risk_label"].unique().tolist()] or sorted(claims_df["risk_label"].unique())

    selected_providers = st.multiselect("Providers", provider_options, default=[])
    selected_risks = st.multiselect("Risk groups", risk_options, default=[])

    min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

    amount_min = float(claims_df["amount_billed"].min()) if len(claims_df) else 0.0
    amount_max = float(claims_df["amount_billed"].max()) if len(claims_df) else 1.0
    if amount_min == amount_max:
        amount_max = amount_min + 1.0
    amount_range = st.slider("Billed amount range", amount_min, amount_max, (amount_min, amount_max))

    st.divider()

    use_gpt4omini = st.toggle("Use GPT-4o-mini", value=True)
    top_k = st.slider("Evidence chunks", 1, 6, 3)

    st.divider()
    st.caption("Run command")
    st.code("streamlit run app/streamlit_app.py", language="bash")


filtered = apply_filters(claims_df, selected_providers, selected_risks, min_risk, amount_range)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_dashboard, tab_claims, tab_provider, tab_docs, tab_assistant, tab_trust = st.tabs(
    [
        "Dashboard",
        "Claims Explorer",
        "Provider Drill-Down",
        "Documents & RAG",
        "Production Assistant",
        "Trust & Validation",
    ]
)


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
with tab_dashboard:
    st.subheader("Executive Risk Dashboard")

    total_claims = len(filtered)
    total_providers = filtered["provider_id"].nunique()
    total_billed = filtered["amount_billed"].sum()
    total_paid = filtered["amount_paid"].sum()
    high_risk_count = int((filtered["risk_label"] == "High").sum())
    high_risk_rate = high_risk_count / total_claims if total_claims else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Filtered Claims", f"{total_claims:,}", "after sidebar filters")
    with c2:
        metric_card("Providers", f"{total_providers:,}", "unique providers")
    with c3:
        metric_card("Total Billed", fmt_money(total_billed), "submitted charges")
    with c4:
        metric_card("Total Paid", fmt_money(total_paid), "paid amount")
    with c5:
        metric_card("High-Risk Rate", f"{high_risk_rate:.1%}", f"{high_risk_count:,} high-risk claims")

    st.markdown("")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Claim Risk Distribution")
        if len(filtered):
            risk_counts = filtered["risk_label"].value_counts().reindex(["High", "Medium", "Low"]).dropna()
            st.bar_chart(risk_counts)

            st.caption(
                "Risk Distribution shows how many claims fall into each AI-assisted risk category "
                "(High, Medium, Low). Higher-risk claims may require manual audit review."
            )
        else:
            st.info("No claims match the selected filters.")

    with right:
        st.markdown("### Average Billed Amount by Risk")
        if len(filtered):
            risk_billed = (
                filtered.groupby("risk_label")["amount_billed"]
                .mean()
                .reindex(["High", "Medium", "Low"])
                .dropna()
            )
            st.bar_chart(risk_billed)
        else:
            st.info("No claims match the selected filters.")

    st.markdown("### Highest-Risk Claims")
    high_priority = filtered.sort_values(["risk_score", "amount_billed"], ascending=False).head(15)
    st.dataframe(high_priority[top_columns(high_priority)], use_container_width=True, height=360)


# ------------------------------------------------------------
# Claims Explorer
# ------------------------------------------------------------
with tab_claims:
    st.subheader("Claims Explorer")
    st.caption("Use filters in the sidebar to quickly inspect risk, billing, procedure, and diagnosis patterns.")

    st.dataframe(filtered[top_columns(filtered)], use_container_width=True, height=360)

    a, b, c = st.columns(3)

    with a:
        st.markdown("### Procedure Mix")
        if len(filtered):
            st.bar_chart(filtered["procedure_code"].astype(str).value_counts().head(12))
        else:
            st.info("No data.")

    with b:
        st.markdown("### Diagnosis Mix")
        if len(filtered):
            st.bar_chart(filtered["diagnosis_code"].astype(str).value_counts().head(12))
        else:
            st.info("No data.")

    with c:
        st.markdown("### Risk Score Summary")
        if len(filtered):
            st.dataframe(filtered[["risk_score", "amount_billed", "amount_paid"]].describe().T, use_container_width=True)
        else:
            st.info("No data.")


# ------------------------------------------------------------
# Active Provider Drill-Down
# ------------------------------------------------------------
with tab_provider:
    st.subheader("Active Provider Drill-Down")
    st.caption("Select a provider to inspect billing behavior, risk profile, procedure mix, and high-risk claims.")

    provider_summary = provider_risk_summary(filtered)

    if provider_summary.empty:
        st.warning("Provider drill-down needs provider_id and amount_billed columns.")
    else:
        # Ensure useful score column exists in summary
        if "avg_risk_score" not in provider_summary.columns and "risk_score" in filtered.columns:
            risk_summary = filtered.groupby("provider_id")["risk_score"].mean().reset_index(name="avg_risk_score")
            provider_summary = provider_summary.merge(risk_summary, on="provider_id", how="left")

        provider_summary = provider_summary.sort_values(
            ["avg_risk_score" if "avg_risk_score" in provider_summary.columns else "total_billed"],
            ascending=False,
        )

        st.markdown("### Provider Ranking")
        st.dataframe(provider_summary.head(50), use_container_width=True, height=300)

        provider_ids = provider_summary["provider_id"].astype(str).tolist()
        default_provider = provider_ids[0] if provider_ids else None

        selected_provider = st.selectbox(
            "Choose provider for detailed review",
            provider_ids,
            index=0 if default_provider else None,
        )

        pclaims = filtered[filtered["provider_id"].astype(str) == str(selected_provider)].copy()

        if len(pclaims) == 0:
            st.warning("No claims available for selected provider after current filters.")
        else:
            p_total_billed = pclaims["amount_billed"].sum()
            p_total_paid = pclaims["amount_paid"].sum()
            p_avg_risk = pclaims["risk_score"].mean()
            p_high_risk = int((pclaims["risk_label"] == "High").sum())

            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                metric_card("Provider", str(selected_provider), "selected provider")
            with p2:
                metric_card("Claims", f"{len(pclaims):,}", "filtered claims")
            with p3:
                metric_card("Total Billed", fmt_money(p_total_billed), "provider charges")
            with p4:
                metric_card("Avg Risk", f"{p_avg_risk:.2f}", "mean risk score")
            with p5:
                metric_card("High-Risk Claims", f"{p_high_risk:,}", "needs review")

            left, right = st.columns(2)

            with left:
                st.markdown("### Provider Procedure Mix")
                st.bar_chart(pclaims["procedure_code"].astype(str).value_counts().head(10))

            with right:
                st.markdown("### Provider Risk Distribution")
                st.bar_chart(pclaims["risk_label"].value_counts().reindex(["High", "Medium", "Low"]).dropna())

            st.markdown("### Provider Claims for Manual Review")
            st.dataframe(
                pclaims.sort_values(["risk_score", "amount_billed"], ascending=False)[top_columns(pclaims)],
                use_container_width=True,
                height=360,
            )

            st.markdown("### Suggested Provider Review Summary")
            st.write(
                f"Provider **{selected_provider}** has **{len(pclaims):,}** filtered claims, "
                f"with average risk score **{p_avg_risk:.2f}** and **{p_high_risk:,}** high-risk claims. "
                "Review should focus on high-billed outliers, repeated procedure patterns, "
                "diagnosis-procedure consistency, and documentation sufficiency."
            )


# ------------------------------------------------------------
# Documents & RAG
# ------------------------------------------------------------
with tab_docs:
    st.subheader("Documents & RAG Evidence Layer")
    st.caption("This tab shows the document/policy layer that grounds the production assistant.")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        metric_card("Clinical Notes", f"{len(notes_df):,}", "available notes")
    with d2:
        metric_card("Policy Text", f"{len(policy_text):,}", "characters")
    with d3:
        metric_card("Audit Rules", f"{len(audit_rules) if isinstance(audit_rules, list) else 1:,}", "rules/context")
    with d4:
        metric_card("Top-K Evidence", str(top_k), "chunks to LLM")

    st.markdown("### Clinical Notes Search")
    search_text = st.text_input("Search clinical notes", "")
    notes_view = notes_df.copy()
    if search_text.strip() and len(notes_view):
        object_cols = [c for c in notes_view.columns if notes_view[c].dtype == "object"]
        mask = pd.Series(False, index=notes_view.index)
        for col in object_cols:
            mask = mask | notes_view[col].astype(str).str.contains(search_text, case=False, na=False)
        notes_view = notes_view[mask]
    st.dataframe(notes_view.head(150), use_container_width=True, height=300)

    col_policy, col_rules = st.columns([1.2, 1])
    with col_policy:
        st.markdown("### CMS / FWA Policy Text")
        st.text_area("Policy context used by RAG", policy_text[:8000], height=320)

    with col_rules:
        st.markdown("### Audit Rules")
        st.json(audit_rules)


# ------------------------------------------------------------
# Production Assistant
# ------------------------------------------------------------
with tab_assistant:
    st.subheader("Production RAG + GPT-4o-mini Assistant")
    st.caption("Grounded answer generation with evidence retrieval, reranking, validation, and hallucination checks.")

    examples = [
        "Why is a claim with abnormal billing and diagnosis-procedure inconsistency suspicious?",
        "What evidence is needed before escalating a provider for manual audit review?",
        "How should duplicate billing patterns be reviewed in healthcare FWA analytics?",
        "How does RAG reduce hallucination in CMS FWA document analysis?",
    ]

    example = st.selectbox("Example question", examples)
    query = st.text_area("Question", value=example, height=120)

    if st.button("Run Production Analysis", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Running production workflow..."):
                result = run_production_analysis(
                    query=query,
                    top_k=top_k,
                    use_gpt4omini=use_gpt4omini,
                )

            if result.get("status") != "success":
                st.error(result.get("message", "Analysis failed."))
            else:
                st.session_state["last_production_result"] = result

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    metric_card("OpenAI Ready", "Yes" if result.get("openai_ready") else "No", "API status")
                with r2:
                    metric_card("Grounding", str(result["validation"].get("grounding_score", "N/A")), "validation score")
                with r3:
                    metric_card("Hallucination", str(result["hallucination"].get("risk", "N/A")), "risk level")
                with r4:
                    metric_card("Faithfulness", str(result["judge"].get("faithfulness", "N/A")), "judge score")

                st.markdown("### Grounded Response")
                st.write(result["answer"])

                with st.expander("Retrieved and reranked evidence"):
                    st.text_area("Grounded context passed to GPT-4o-mini", result["grounded_context"], height=280)

                with st.expander("Raw production workflow output"):
                    st.json(result)


# ------------------------------------------------------------
# Trust & Validation
# ------------------------------------------------------------
with tab_trust:
    st.subheader("Trust, Validation, and Hallucination Reduction")

    result = st.session_state.get("last_production_result")

    if not result:
        st.info("Run the Production Assistant first to populate validation results.")
    else:
        t1, t2, t3 = st.columns(3)
        with t1:
            metric_card("Grounding Score", str(result["validation"].get("grounding_score", "N/A")), "higher is better")
        with t2:
            metric_card("Hallucination Score", str(result["hallucination"].get("score", "N/A")), "lower is better")
        with t3:
            metric_card("Judge Groundedness", str(result["judge"].get("groundedness", "N/A")), "evaluation signal")

        col_val, col_hal = st.columns(2)
        with col_val:
            st.markdown("### Validation Layer")
            st.json(result["validation"])
        with col_hal:
            st.markdown("### Hallucination Scoring")
            st.json(result["hallucination"])

        st.markdown("### Judge Evaluation")
        st.json(result["judge"])

    st.markdown("### Final Production Architecture")
    st.code(
        """
Claims + Documents
    ↓
Dashboard Analytics
    ↓
RAG Retriever
    ↓
Reranker
    ↓
Grounded Context
    ↓
GPT-4o-mini Reasoning
    ↓
Validation + Hallucination Scoring
    ↓
Judge Evaluation
    ↓
Final AI-Assisted Review Recommendation
        """,
        language="text",
    )
