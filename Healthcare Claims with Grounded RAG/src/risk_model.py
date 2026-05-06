import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


class FWARiskModel:
    """
    Supervised ML model for FWA risk scoring.

    Annotation:
    This model learns from labeled examples and produces a risk score. A score is
    more useful in compliance workflows than only a 0/1 label.
    """

    def __init__(self, preprocessor):
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", XGBClassifier(
                    n_estimators=60,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                )),
            ]
        )

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        pred = self.pipeline.predict(X_test)
        prob = self.pipeline.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, prob),
        }

    def predict(self, claim_df):
        score = float(self.pipeline.predict_proba(claim_df)[:, 1][0])

        if score >= HIGH_RISK_THRESHOLD:
            level = "High"
        elif score >= MEDIUM_RISK_THRESHOLD:
            level = "Medium"
        else:
            level = "Low"

        return {"risk_score": round(score, 3), "risk_level": level}

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path):
        model = FWARiskModel.__new__(FWARiskModel)
        model.pipeline = joblib.load(path)
        return model
