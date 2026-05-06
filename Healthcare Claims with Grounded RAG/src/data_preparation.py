import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_claims(path):
    """
    Load claims data and validate schema.

    Annotation:
    Schema validation is a critical production step. Without it, a missing column
    or changed upstream data feed can silently break model inference.
    """
    df = pd.read_csv(path)

    required_cols = [
        "claim_id", "provider_id", "patient_id", "procedure_code", "diagnosis_code",
        "claim_amount", "patient_age", "num_prior_claims", "days_since_last_claim",
        "is_high_risk_provider", "label"
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def build_preprocessor():
    """
    Build reusable preprocessing logic.

    Annotation:
    Training and inference must use the same preprocessing. A Pipeline prevents
    training-serving skew.
    """
    numeric_features = [
        "claim_amount", "patient_age", "num_prior_claims",
        "days_since_last_claim", "is_high_risk_provider"
    ]

    categorical_features = ["provider_id", "procedure_code", "diagnosis_code"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def create_train_test(df):
    X = df.drop(columns=["label", "claim_id", "patient_id"])
    y = df["label"]

    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
