
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "insurance_diabetes.csv"


def generate_synthetic_data(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic diabetes-related medical insurance cost data."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 86, n_rows)
    sex = rng.choice(["male", "female"], n_rows)
    bmi = np.clip(rng.normal(29, 6, n_rows), 16, 55).round(1)
    smoker = rng.choice(["yes", "no"], n_rows, p=[0.2, 0.8])
    children = rng.integers(0, 6, n_rows)
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n_rows)

    diabetes_prob = 1 / (1 + np.exp(-(-6 + 0.045 * age + 0.09 * bmi + (smoker == "yes") * 0.35)))
    diabetes = np.where(rng.random(n_rows) < diabetes_prob, "yes", "no")

    hypertension_prob = 1 / (1 + np.exp(-(-5.5 + 0.055 * age + 0.075 * bmi + (diabetes == "yes") * 0.9)))
    hypertension = np.where(rng.random(n_rows) < hypertension_prob, "yes", "no")

    hba1c = np.where(diabetes == "yes", rng.normal(7.7, 1.4, n_rows), rng.normal(5.4, 0.45, n_rows))
    hba1c = np.clip(hba1c, 4.5, 13.5).round(1)

    region_adj = np.select(
        [region == "southeast", region == "northeast", region == "northwest", region == "southwest"],
        [900, 700, 500, 400],
    )

    cost = 2500 + age * 95 + bmi * 210 + children * 450 + region_adj
    cost += (smoker == "yes") * 11000
    cost += (diabetes == "yes") * 6500
    cost += (hypertension == "yes") * 4200
    cost += np.maximum(hba1c - 6, 0) * 1800
    cost += rng.normal(0, 2200, n_rows)
    cost = np.clip(cost, 2500, None).round(2)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "smoker": smoker,
            "children": children,
            "region": region,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "hba1c": hba1c,
            "annual_claim_cost": cost,
        }
    )


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")
