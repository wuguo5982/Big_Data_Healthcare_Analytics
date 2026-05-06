import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


class FWAAnomalyModel:
    """
    Unsupervised anomaly model for unusual billing behavior.

    Annotation:
    This complements supervised ML. Many suspicious billing patterns may not be
    labeled, so anomaly detection can catch rare or emerging patterns.
    """

    def __init__(self, preprocessor):
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", IsolationForest(
                    n_estimators=80,
                    contamination=0.18,
                    random_state=42,
                )),
            ]
        )

    def train(self, X_train):
        self.pipeline.fit(X_train)

    def predict(self, claim_df: pd.DataFrame):
        pred = self.pipeline.predict(claim_df)[0]
        raw_score = self.pipeline.decision_function(claim_df)[0]
        return {"is_anomaly": bool(pred == -1), "anomaly_score": round(float(raw_score), 3)}

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path):
        model = FWAAnomalyModel.__new__(FWAAnomalyModel)
        model.pipeline = joblib.load(path)
        return model
