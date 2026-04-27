import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
MODEL_PATH = ROOT_DIR / "models" / "insurance_diabetes_linear_model.json"

if not MODEL_PATH.exists():
    subprocess.run([sys.executable, str(ROOT_DIR / "training" / "train_model.py")], check=True)

from tools.insurance_prediction_tool import predict_insurance_cost


def test_prediction_tool():
    result = predict_insurance_cost(
        age=58,
        sex="male",
        bmi=32.5,
        smoker="yes",
        children=1,
        region="southeast",
        diabetes="yes",
        hypertension="yes",
        hba1c=8.2,
    )
    assert result["estimated_annual_cost_usd"] > 0
    assert result["risk_level"] in {"Low", "Moderate", "High", "Very High"}
    assert "disclaimer" in result
    print(result)


if __name__ == "__main__":
    test_prediction_tool()
