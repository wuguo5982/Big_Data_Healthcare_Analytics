# Clear EDA and minimal SMOTE modeling update

Run `notebooks/01_EDA.ipynb` first, then `notebooks/02_Modeling.ipynb`, from top to bottom. The EDA notebook has been lightly organized into clear sections, with consistent variable names and the original feature definitions.

EDA changes: merge repeated checks and cleaning, use train_df/test_df and X/y/X_test consistently, validate joins, and retain the original provider-feature function. Label-based EDA uses the same 75% training partition as modeling.

Modeling changes: add training-only StandardScaler + SMOTE; use the balanced data for the existing Logistic Regression, Random Forest and XGBoost models; add KNN and SVM; compare accuracy, precision, recall, F1, average precision and ROC AUC using the original 25% validation split and a 0.5 threshold.

The notebook includes executed outputs. Its comparison is saved to `artifacts/reports/smote_model_comparison.csv`. This is a validation comparison, not an independent final test result. Earlier result/model files and the original dashboard are unchanged and do not represent the new SMOTE experiment.

Install notebook dependencies with `python -m pip install -r requirements.txt`. Tested SMOTE experiment: Python 3.12, scikit-learn 1.8.0, imbalanced-learn 0.14.2, XGBoost 3.4.1. The original project requirements use scikit-learn 1.7; retrain locally rather than loading old model files across versions.

---


# CMS Provider Fraud Detection

This portable version trains the Logistic Regression model on your own computer.
It does not include pre-trained `.joblib` files, so there is no scikit-learn
version mismatch.

## Requirements

- Python 3.10
- Conda or another virtual environment

## Project structure

```text
CMS_Fraud_Project_Portable_Final/
├── app/
│   ├── app.py
│   └── examples/
├── artifacts/
│   ├── models/
│   ├── predictions/
│   └── reports/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── src/
│   ├── config.py
│   ├── llm_explainer.py
│   ├── predict.py
│   └── train_logistic.py
├── .env.example
├── README.md
└── requirements.txt
```

## 1. Activate the environment

```bash
conda activate medical-ai
```

## 2. Install packages

```bash
python -m pip install -r requirements.txt
```

## 3. Train the deployment model locally

```bash
python src/train_logistic.py
```

This creates:

```text
artifacts/models/logistic_model.joblib
artifacts/models/standard_scaler.joblib
artifacts/models/model_metadata.json
artifacts/models/logistic_coefficients.csv
```

Because these files are created in your own environment, they match your
installed scikit-learn version.

## 4. Configure the OpenAI explanation

Copy `.env.example` to `.env`:

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

The prediction works without an OpenAI API key. The key is used only to explain
flagged cases.

## 5. Run the Streamlit app

From the project root:

```bash
python -m streamlit run app/app.py
```

## App workflow

```text
Enter or upload a provider profile
→ Logistic Regression predicts fraud probability
→ app shows fraud risk and strongest model contributions
→ LLM explains flagged cases using only those grounded model signals
```

## Important limitation

This is a provider-level screening model based on aggregated claim features.
A flagged result is not proof of fraud and requires human review.
