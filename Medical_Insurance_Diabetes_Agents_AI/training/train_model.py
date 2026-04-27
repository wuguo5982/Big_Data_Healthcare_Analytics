from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "insurance_diabetes.csv"
MODEL_PATH = ROOT_DIR / "models" / "insurance_diabetes_linear_model.json"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

NUMERIC_FEATURES = ["age", "bmi", "children", "hba1c"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region", "diabetes", "hypertension"]
TARGET = "annual_claim_cost"

CATEGORY_LEVELS = {
    "sex": ["male", "female"],
    "smoker": ["yes", "no"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
    "diabetes": ["yes", "no"],
    "hypertension": ["yes", "no"],
}


def build_design_matrix(df: pd.DataFrame, include_intercept: bool = True):
    """Create a simple numeric + one-hot design matrix without sklearn."""
    columns = []
    arrays = []

    if include_intercept:
        columns.append("intercept")
        arrays.append(np.ones(len(df)))

    for col in NUMERIC_FEATURES:
        columns.append(col)
        arrays.append(df[col].astype(float).to_numpy())

    # Drop the last category for each variable to avoid perfect collinearity.
    for col, levels in CATEGORY_LEVELS.items():
        values = df[col].astype(str).str.lower().str.strip()
        for level in levels[:-1]:
            columns.append(f"{col}__{level}")
            arrays.append((values == level).astype(float).to_numpy())

    X = np.vstack(arrays).T
    return X, columns


def train_model() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Deterministic train/test split without sklearn.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split]
    test_df = shuffled.iloc[split:]

    X_train, feature_names = build_design_matrix(train_df)
    y_train = train_df[TARGET].astype(float).to_numpy()

    # Ridge-stabilized linear regression: beta = (X'X + lambda I)^-1 X'y
    ridge_lambda = 1.0
    identity = np.eye(X_train.shape[1])
    identity[0, 0] = 0.0  # do not penalize intercept
    beta = np.linalg.solve(X_train.T @ X_train + ridge_lambda * identity, X_train.T @ y_train)

    X_test, _ = build_design_matrix(test_df)
    y_test = test_df[TARGET].astype(float).to_numpy()
    pred = X_test @ beta

    mae = float(np.mean(np.abs(y_test - pred)))
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    model = {
        "model_type": "RidgeLinearRegressionFromScratch",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_levels": CATEGORY_LEVELS,
        "feature_names": feature_names,
        "coefficients": [float(x) for x in beta],
    }

    metrics = {
        "model_type": model["model_type"],
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train_model()
