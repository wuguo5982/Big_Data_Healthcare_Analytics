# Medical Insurance Cost Prediction with MLflow and SHAP

# This project implements an advanced machine learning pipeline to predict medical insurance charges
# using regression techniques. It includes preprocessing (standardization, polynomial expansion),
# hyperparameter tuning (GridSearchCV), performance evaluation, and model tracking using MLflow.
# The model explainability is enhanced via SHAP to better understand feature contributions.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise e

# Encode categorical variables
df = load_data()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Advanced_Medical_Cost_Prediction")

# Define pipeline with polynomial features
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("regressor", Ridge(alpha=1.0))
])

# Hyperparameter grid
param_grid = {
    "regressor__alpha": [0.1, 1.0, 10.0]
}

# Grid search with cross-validation
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

# Enable autologging
mlflow.sklearn.autolog()


def main(alpha, l1_ratio):
    with mlflow.start_run():
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=X_test.iloc[:2])

        # SHAP explainability
        explainer = shap.Explainer(best_model.named_steps["regressor"], best_model.named_steps["poly"].transform(X_train))
        shap_values = explainer(best_model.named_steps["poly"].transform(X_test))
        plt.figure()
        shap.summary_plot(shap_values, best_model.named_steps["poly"].transform(X_test), show=False)
        plt.savefig("shap_summary_plot.png")
        mlflow.log_artifact("shap_summary_plot.png")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type=float, default=0.3)
    args.add_argument("--l1_ratio","-l1", type=float, default=0.5)
    parsed_args = args.parse_args()
    
    # parsed_args.param1
    main(parsed_args.alpha, parsed_args.l1_ratio)


# 1. Summary / Conclusion:

# This project successfully implemented a comprehensive ML pipeline to predict individual medical insurance costs based on demographic and health-related attributes. 
# By integrating Ridge regression, polynomial feature expansion, and hyperparameter optimization (GridSearchCV), the model achieved solid predictive performance. 
# All stages of the experiment — including parameters, metrics, artifacts, and SHAP-based model explanations — 
# were efficiently tracked and visualized using MLflow, ensuring reproducibility and traceability.

# The final model demonstrated:

# (a). High interpretability using SHAP plots

# (b). Consistent tracking of performance metrics (MAE, RMSE, R²)

# (c). A modular design, easily extendable for production deployment


# Potential Improvements:
# (1). Model Complexity: Try other regression models (e.g., XGBoost, LightGBM) to capture complex patterns more effectively.

# (2). Feature Engineering: Incorporate domain-specific engineered features or binning strategies (e.g., age groups, BMI categories).

# (3). Hyperparameter Tuning: Use RandomizedSearchCV or Bayesian optimization for faster or smarter parameter searches.

# (4). Cross-Validation: Use KFold cross-validation scoring on the entire dataset for better generalization estimates.

# (5). Pipeline Deployment: Package the best model into a REST API using MLflow Models + Flask/FastAPI, or deploy via MLflow’s built-in model serving.
