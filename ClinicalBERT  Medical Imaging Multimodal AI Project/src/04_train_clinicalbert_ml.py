"""Train ML models using structured EHR data + ClinicalBERT embeddings."""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, project_root


def evaluate_model(y_true: pd.Series, y_probability: np.ndarray) -> dict:
    """
    Evaluate model performance.

    y_probability is the predicted probability for the positive class.
    If probability >= 0.5, we predict high risk.
    """

    y_pred = (y_probability >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(y_true, y_probability),
    }


def load_training_data(config: dict, root) -> pd.DataFrame:
    """
    Load structured EHR data and ClinicalBERT embeddings,
    then merge them by patient_id.
    """

    ehr_data = pd.read_csv(root / config["data"]["structured_ehr"])
    patient_index = pd.read_csv(root / config["data"]["patient_index"])
    bert_embeddings = np.load(root / config["data"]["clinicalbert_embeddings"])

    bert_columns = [
        f"bert_{i}"
        for i in range(bert_embeddings.shape[1])
    ]

    bert_df = pd.DataFrame(bert_embeddings, columns=bert_columns)
    bert_df["patient_id"] = patient_index["patient_id"]

    training_data = ehr_data.merge(bert_df, on="patient_id")

    return training_data


def split_features_and_label(df: pd.DataFrame):
    """
    Separate input features X from target label y.
    """

    columns_to_remove = [
        "patient_id",
        "encounter_id",
        "image_id",
        "primary_condition",
        "high_risk_label",
    ]

    X = df.drop(columns=columns_to_remove)
    y = df["high_risk_label"]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing steps:
    - Scale numeric columns
    - One-hot encode categorical columns
    """

    categorical_columns = [
        column
        for column in X.columns
        if X[column].dtype == "object"
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    return preprocessor


def build_models() -> dict:
    """
    Create candidate ML models.
    """

    return {
        "logistic_regression_clinicalbert": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest_clinicalbert": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "extra_trees_clinicalbert": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def train_and_compare_models(X, y, preprocessor):
    """
    Train multiple models and select the best one by ROC-AUC.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    models = build_models()

    results = []
    best_model = None
    best_roc_auc = -1

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_probability = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_probability)
        metrics["model"] = model_name
        results.append(metrics)

        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roc_auc", ascending=False)

    return results_df, best_model


def save_results(results_df: pd.DataFrame, best_model, root) -> None:
    """
    Save model comparison results and the best trained model.
    """

    output_dir = root / "outputs"
    model_dir = root / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "clinicalbert_model_comparison.csv"
    model_path = model_dir / "best_clinicalbert_model.pkl"

    results_df.to_csv(results_path, index=False)
    joblib.dump(best_model, model_path)

    print("Saved model comparison to:", results_path)
    print("Saved best model to:", model_path)


def main() -> None:
    """
    Main workflow:
    1. Load EHR data
    2. Load ClinicalBERT embeddings
    3. Merge data by patient_id
    4. Train several ML models
    5. Save model comparison and best model
    """

    config = load_config()
    root = project_root()

    training_data = load_training_data(config, root)

    X, y = split_features_and_label(training_data)

    preprocessor = build_preprocessor(X)

    results_df, best_model = train_and_compare_models(
        X=X,
        y=y,
        preprocessor=preprocessor,
    )

    save_results(results_df, best_model, root)

    print(results_df)


if __name__ == "__main__":
    main()