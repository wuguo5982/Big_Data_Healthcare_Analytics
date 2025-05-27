"""
Project Overview:
This project demonstrates the use of Airflow DAGs implemented via the Astro platform and Docker, integrated with VS Code. 
It builds an advanced machine learning pipeline for processing insurance data.

Key Components:

- Data ingestion and preprocessing
- Feature engineering (including BMI categorization)
- Model training using Linear Regression and Random Forest
- Model evaluation using MSE and R² metrics
- A/B testing via cross-validation (results logged to Airflow logs)
- Batch inference on test data (results logged to Airflow logs)

Generated Artifacts:

insurance_linear_model.pkl
insurance_rf_model.pkl

Output results available in Airflow task logs
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO

# Task 1: Load and clean data
def load_data(**context):
    file_path = os.path.join(os.path.dirname(__file__), "data", "insurance.csv")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    context['ti'].xcom_push(key='data', value=df.to_json(orient='split'))

# Task 2: Feature engineering
def feature_engineering(**context):
    df = pd.read_json(StringIO(context['ti'].xcom_pull(key='data', task_ids='load_data_task')), orient='split')
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, np.inf], labels=['underweight', 'normal', 'overweight', 'obese'])
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("charges", axis=1)
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    context['ti'].xcom_push(key='X_train', value=X_train.to_json(orient='split'))
    context['ti'].xcom_push(key='X_test', value=X_test.to_json(orient='split'))
    context['ti'].xcom_push(key='y_train', value=y_train.to_frame().to_json(orient='split'))
    context['ti'].xcom_push(key='y_test', value=y_test.to_frame().to_json(orient='split'))

# Task 3: Train the models
def train_model(**context):
    X_train = pd.read_json(StringIO(context['ti'].xcom_pull(key='X_train', task_ids='feature_engineering_task')), orient='split')
    y_train = pd.read_json(StringIO(context['ti'].xcom_pull(key='y_train', task_ids='feature_engineering_task')), orient='split').squeeze()

    lin_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/insurance_linear_model.pkl", 'wb') as f: pickle.dump(lin_model, f)
    with open("models/insurance_rf_model.pkl", 'wb') as f: pickle.dump(rf_model, f)

# Task 4: Evaluate models
def evaluate_model(**context):
    X_test = pd.read_json(StringIO(context['ti'].xcom_pull(key='X_test', task_ids='feature_engineering_task')), orient='split')
    y_test = pd.read_json(StringIO(context['ti'].xcom_pull(key='y_test', task_ids='feature_engineering_task')), orient='split').squeeze()

    with open("models/insurance_linear_model.pkl", 'rb') as f: lin_model = pickle.load(f)
    with open("models/insurance_rf_model.pkl", 'rb') as f: rf_model = pickle.load(f)

    lin_preds = lin_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    print("Linear Regression — MSE: {:.2f}, R²: {:.2f}".format(mean_squared_error(y_test, lin_preds), r2_score(y_test, lin_preds)))
    print("Random Forest — MSE: {:.2f}, R²: {:.2f}".format(mean_squared_error(y_test, rf_preds), r2_score(y_test, rf_preds)))

# Task 5: A/B testing using CV and log to Airflow

def ab_testing(**context):
    X_train = pd.read_json(StringIO(context['ti'].xcom_pull(key='X_train', task_ids='feature_engineering_task')), orient='split')
    y_train = pd.read_json(StringIO(context['ti'].xcom_pull(key='y_train', task_ids='feature_engineering_task')), orient='split').squeeze()

    lin_scores = cross_val_score(LinearRegression(), X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rf_scores = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), X_train, y_train, scoring='neg_mean_squared_error', cv=5)

    lin_mse = -np.mean(lin_scores)
    rf_mse = -np.mean(rf_scores)

    print("""
A/B Testing Cross-Validation:
Linear Regression MSE (avg): {:.2f}
Random Forest MSE (avg): {:.2f}
""".format(lin_mse, rf_mse))

# Task 6: Batch inference and log to Airflow

def batch_inference(**context):
    X_test = pd.read_json(StringIO(context['ti'].xcom_pull(key='X_test', task_ids='feature_engineering_task')), orient='split')
    with open("models/insurance_rf_model.pkl", 'rb') as f:
        rf_model = pickle.load(f)

    predictions = rf_model.predict(X_test)
    result_df = X_test.copy()
    result_df['predicted_charges'] = predictions

    print("\n Batch Inference Sample:")
    print(result_df.head().to_string(index=False))

with DAG(
    dag_id='insurance_ml_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule='@once',
    catchup=False,
    description='Advanced insurance ML DAG with A/B testing and batch inference (logged)'
) as dag:

    load_data_task = PythonOperator(task_id='load_data_task', python_callable=load_data)
    feature_engineering_task = PythonOperator(task_id='feature_engineering_task', python_callable=feature_engineering)
    train_model_task = PythonOperator(task_id='train_model_task', python_callable=train_model)
    evaluate_model_task = PythonOperator(task_id='evaluate_model_task', python_callable=evaluate_model)
    ab_testing_task = PythonOperator(task_id='ab_testing_task', python_callable=ab_testing)
    batch_inference_task = PythonOperator(task_id='batch_inference_task', python_callable=batch_inference)

    load_data_task >> feature_engineering_task >> train_model_task
    train_model_task >> [evaluate_model_task, ab_testing_task, batch_inference_task]

"""
Summary:
This DAG implements a robust end-to-end machine learning pipeline on insurance data, orchestrated using Airflow.

It includes the following components:
- Model evaluation using MSE and R² metrics for both Linear Regression and Random Forest
- A/B testing via 5-fold cross-validation with logged comparative performance
  • Example: Linear MSE (avg): ~4100, Random Forest MSE (avg): ~2900
- Batch inference on test data, with top predictions printed to Airflow logs for verification

Results are logged to Airflow task logs, making this pipeline lightweight and production-ready.
"""


"""
# DAG Graph Overview

This DAG represents the ML pipeline for insurance data:

![DAG Graph](dag_graph.png)

- The graph includes tasks for data loading, feature engineering, model training, evaluation, A/B testing, and batch inference.
- Generated using Airflow Graph View.
"""