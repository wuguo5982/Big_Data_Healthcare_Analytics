from pathlib import Path
from src.config import RAW_DIR, MODEL_DIR
from src.generate_sample_data import generate
from src.data_loader import load_claims, load_providers
from src.fraud_models import train_models, score_providers

if __name__ == "__main__":
    if not (RAW_DIR / "claims.csv").exists():
        generate(str(RAW_DIR))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    claims = load_claims(); providers = load_providers()
    obj = train_models(claims, providers, str(MODEL_DIR / "fwa_model.joblib"))
    scores = score_providers(claims, providers, obj)
    scores.to_csv(MODEL_DIR / "provider_risk_scores.csv", index=False)
    print(scores.head(10).to_string(index=False))
