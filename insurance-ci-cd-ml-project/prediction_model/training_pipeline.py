import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from prediction_model.pipeline import create_pipeline

def run_training():
    # Load dataset
    df = pd.read_csv("insurance.csv")
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Split the data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = create_pipeline()
    model.fit(X_train, y_train)

    # Save to correct location
    model_path = os.path.join("prediction_model", "trained_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    run_training()
