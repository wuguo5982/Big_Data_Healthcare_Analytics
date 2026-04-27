from pathlib import Path
import json
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "insurance_diabetes_linear_model.json"
OUTPUT_PATH = ROOT_DIR / "outputs" / "feature_importance.csv"


def save_feature_importance() -> pd.DataFrame:
    """Save coefficient-based feature importance for the from-scratch linear model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train model first: python training/train_model.py")

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    result = pd.DataFrame(
        {
            "feature": model["feature_names"],
            "coefficient": model["coefficients"],
        }
    )
    result["abs_coefficient"] = result["coefficient"].abs()
    result = result[result["feature"] != "intercept"].sort_values("abs_coefficient", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved feature importance to {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    print(save_feature_importance().head(20))
