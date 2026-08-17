# risk_model.py
# Readmission Risk Model encapsulates a machine learning model for predicting 30-day readmission risk based on structured patient, claims, and lab data. 
# It provides methods for training, saving, loading, and predicting risk, along with feature importance for interpretability.

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.feature_engineering import FEATURES, build_training_frame

class ReadmissionRiskModel:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.validation_auc = None

    def train(self, data):
        df = build_training_frame(data)
        X = df[FEATURES]
        y = df["readmission_30d"]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        pred = self.model.predict_proba(X_valid)[:, 1]
        self.validation_auc = float(roc_auc_score(y_valid, pred))

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "validation_auc": self.validation_auc},
            self.model_path
        )
        return self.validation_auc

    def load_or_train(self, data):
        if self.model_path.exists():
            saved = joblib.load(self.model_path)
            self.model = saved["model"]
            self.validation_auc = saved.get("validation_auc")
        else:
            self.train(data)
        return self

    def predict(self, X):
        probability = float(self.model.predict_proba(X)[0, 1])
        importance = dict(zip(FEATURES, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        return probability, top_features
