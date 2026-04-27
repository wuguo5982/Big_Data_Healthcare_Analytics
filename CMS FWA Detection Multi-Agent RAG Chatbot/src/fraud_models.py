from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from .feature_engineering import provider_features, model_matrix

@dataclass
class ModelResult:
    provider_id: str
    anomaly_score: float
    risk_score: float
    risk_level: str
    top_reasons: list[str]

def train_models(claims: pd.DataFrame, providers: pd.DataFrame, model_path: str = "models/fwa_model.joblib"):
    feats = provider_features(claims, providers)
    X, cols = model_matrix(feats)
    y = (feats["synthetic_fwa_labels"] > 0).astype(int)
    iso = Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(n_estimators=200, contamination=0.08, random_state=42))])
    iso.fit(X)
    if y.nunique() > 1:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        clf.fit(X, y)
    else:
        clf = None
    obj = {"isolation_forest": iso, "classifier": clf, "columns": cols}
    joblib.dump(obj, model_path)
    return obj

def score_providers(claims: pd.DataFrame, providers: pd.DataFrame, model_obj=None) -> pd.DataFrame:
    feats = provider_features(claims, providers)
    X, cols = model_matrix(feats)
    if model_obj is None:
        iso = Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(n_estimators=200, contamination=0.08, random_state=42))])
        iso.fit(X)
        model_obj = {"isolation_forest": iso, "classifier": None, "columns": cols}
    iso = model_obj["isolation_forest"]
    raw = -iso.decision_function(X)
    raw_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    feats["anomaly_score"] = raw_norm
    if model_obj.get("classifier") is not None:
        probs = model_obj["classifier"].predict_proba(X)[:, 1]
        feats["ml_probability"] = probs
        feats["risk_score"] = 0.55 * raw_norm + 0.45 * probs
    else:
        feats["ml_probability"] = np.nan
        feats["risk_score"] = raw_norm
    feats["risk_level"] = pd.cut(feats["risk_score"], bins=[-0.01, .35, .7, 1.01], labels=["Low", "Medium", "High"])
    return feats.sort_values("risk_score", ascending=False)

def reasons_for_provider(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("duplicate_rate", 0) > 0.03: reasons.append("Elevated duplicate-claim rate")
    if row.get("dme_ratio", 0) > 0.20: reasons.append("High DME billing ratio")
    if row.get("mismatch_ratio", 0) > 0.08: reasons.append("Diagnosis/procedure medical-necessity mismatch")
    if row.get("upcoding_ratio", 0) > 0.10: reasons.append("Possible high-complexity upcoding pattern")
    if row.get("paid_per_beneficiary", 0) > 1500: reasons.append("High paid amount per beneficiary")
    if row.get("risk_score", 0) > 0.70: reasons.append("High model anomaly risk score")
    return reasons or ["No dominant risk driver found; review supporting claims if needed"]
