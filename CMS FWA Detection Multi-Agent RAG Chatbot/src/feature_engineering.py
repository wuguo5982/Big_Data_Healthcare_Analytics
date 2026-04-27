from __future__ import annotations
import pandas as pd
import numpy as np

def add_claim_flags(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.copy()
    dup_cols = ["provider_id", "beneficiary_id", "service_date", "cpt_hcpcs_code", "paid_amount"]
    df["duplicate_claim_flag"] = df.duplicated(dup_cols, keep=False).astype(int)
    df["high_complexity_low_dx_flag"] = ((df["cpt_hcpcs_code"] == "99215") & (df["icd10_code"].isin(["E11.9", "I10"]))).astype(int)
    df["dme_code_flag"] = df["cpt_hcpcs_code"].isin(["E0260", "E1390"]).astype(int)
    df["dme_dx_mismatch_flag"] = ((df["dme_code_flag"] == 1) & (~df["icd10_code"].isin(["J44.9", "R26.2", "Z51.5"]))).astype(int)
    df["high_units_flag"] = (df["units"] >= df["units"].quantile(0.95)).astype(int)
    df["high_paid_flag"] = (df["paid_amount"] >= df["paid_amount"].quantile(0.95)).astype(int)
    return df

def provider_features(claims: pd.DataFrame, providers: pd.DataFrame | None = None) -> pd.DataFrame:
    df = add_claim_flags(claims)
    agg = df.groupby("provider_id").agg(
        total_claims=("claim_id", "count"),
        total_paid=("paid_amount", "sum"),
        avg_paid=("paid_amount", "mean"),
        avg_units=("units", "mean"),
        duplicate_claims=("duplicate_claim_flag", "sum"),
        dme_claims=("dme_code_flag", "sum"),
        dme_mismatch_claims=("dme_dx_mismatch_flag", "sum"),
        upcoding_flags=("high_complexity_low_dx_flag", "sum"),
        high_paid_claims=("high_paid_flag", "sum"),
        unique_beneficiaries=("beneficiary_id", "nunique"),
        synthetic_fwa_labels=("is_fwa_synthetic_label", "sum"),
    ).reset_index()
    agg["duplicate_rate"] = agg["duplicate_claims"] / agg["total_claims"].clip(lower=1)
    agg["dme_ratio"] = agg["dme_claims"] / agg["total_claims"].clip(lower=1)
    agg["mismatch_ratio"] = agg["dme_mismatch_claims"] / agg["total_claims"].clip(lower=1)
    agg["upcoding_ratio"] = agg["upcoding_flags"] / agg["total_claims"].clip(lower=1)
    agg["paid_per_beneficiary"] = agg["total_paid"] / agg["unique_beneficiaries"].clip(lower=1)
    if providers is not None:
        agg = agg.merge(providers, on="provider_id", how="left")
    return agg

def model_matrix(features: pd.DataFrame):
    cols = [
        "total_claims", "total_paid", "avg_paid", "avg_units", "duplicate_rate",
        "dme_ratio", "mismatch_ratio", "upcoding_ratio", "paid_per_beneficiary", "unique_beneficiaries"
    ]
    X = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, cols
