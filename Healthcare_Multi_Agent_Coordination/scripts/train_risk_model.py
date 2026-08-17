from src.config import DATA_DIR, MODEL_DIR
from src.data_loader import load_data
from src.models.risk_model import ReadmissionRiskModel

data = load_data(DATA_DIR)
model = ReadmissionRiskModel(
    MODEL_DIR / "readmission_random_forest.joblib"
)
auc = model.train(data)
print(f"Validation AUROC: {auc:.3f}")
