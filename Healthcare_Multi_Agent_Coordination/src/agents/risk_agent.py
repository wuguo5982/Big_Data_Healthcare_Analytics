# risk_agent.py
# Risk Agent uses a trained machine learning model to predict 30-day readmission risk based on structured patient, claims, and lab data. 
# It outputs the predicted probability, risk level, and top contributing features for interpretability.

from src.feature_engineering import patient_feature_frame

class RiskAgent:
    name = "risk_agent"

    def __init__(self, risk_model):
        self.risk_model = risk_model

    def run(self, rows: dict) -> dict:
        X = patient_feature_frame(rows["patient"], rows["claim"], rows["lab"])
        probability, top_features = self.risk_model.predict(X)
        level = "HIGH" if probability >= 0.60 else "MODERATE" if probability >= 0.35 else "LOW"
        return {
            "readmission_probability": round(probability, 3),
            "risk_level": level,
            "top_model_features": [
                {"feature": name, "global_importance": round(float(score), 4)}
                for name, score in top_features
            ],
            "validation_auc": (
                round(float(self.risk_model.validation_auc), 3)
                if self.risk_model.validation_auc is not None else None
            ),
        }
