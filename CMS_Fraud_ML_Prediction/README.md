# Healthcare Provider Fraud Risk Detection with Machine Learning

## Introduction

This project analyzes healthcare claims and beneficiary data to identify providers at risk of potential fraud.

Provider-level exploratory data analysis (EDA) examines data quality, class imbalance, claim volume, reimbursement patterns, and feature relationships through seven annotated visualizations. The analysis creates 33 provider features for modeling.

Five models are compared: Logistic Regression, Random Forest, XGBoost, KNN, and SVM. A stratified 75/25 training–validation split is used. Scaling is fitted on training data, and SMOTE is applied only to the training set.

## Results

| Model | Accuracy | Precision | Recall | F1 | Average Precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 90.4% | 49.3% | 86.6% | 0.629 | 0.758 |
| Random Forest | 92.9% | 59.2% | 78.7% | 0.676 | 0.701 |
| XGBoost | 93.3% | 61.6% | 77.2% | 0.685 | 0.744 |
| KNN | 87.4% | 41.7% | 86.6% | 0.563 | 0.495 |
| SVM | 90.8% | 50.5% | 86.6% | 0.638 | 0.603 |

- **Logistic Regression** achieved the highest average precision (0.758), with 86.6% recall.
- **XGBoost** achieved the highest accuracy (93.3%), precision (61.6%), and F1 score (0.685).

Precision, recall, and F1 refer to the positive class at a 0.5 threshold. Predictions flag providers for review; they do not establish fraud. Independent labeled data is needed for final evaluation.

Note: All the data from Kaggle.
