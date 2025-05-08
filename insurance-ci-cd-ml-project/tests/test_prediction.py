import joblib
import pandas as pd
import os
import sys
import pytest

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Path to the saved model
# MODEL_PATH = os.path.join(os.getcwd(), "trained_model.joblib")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_model.joblib"))


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), "Model file not found. Run training_pipeline.py first."
    return joblib.load(MODEL_PATH)

def test_insurance_model_prediction(model):
    # Sample input for prediction
    sample_input = {
        "age": 45,
        "sex": "female",
        "bmi": 24.0,
        "children": 1,
        "smoker": "no",
        "region": "southeast"
    }

    input_df = pd.DataFrame([sample_input])
    prediction = model.predict(input_df)

    assert prediction is not None
    assert isinstance(prediction[0], float)
    assert prediction[0] > 0  # Insurance charges should be positive
