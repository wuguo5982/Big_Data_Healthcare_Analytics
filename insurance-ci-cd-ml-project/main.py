import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel

# Load the full pipeline model
MODEL_PATH = os.path.join("prediction_model", "trained_model.joblib")
model = joblib.load(MODEL_PATH)

# Flask app for HTML + form submission
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        form = request.form
        input_dict = {
            "age": int(form["age"]),
            "sex": form["sex"],
            "bmi": float(form["bmi"]),
            "children": int(form["children"]),
            "smoker": form["smoker"],
            "region": form["region"]
        }

        df = pd.DataFrame([input_dict])
        prediction = model.predict(df)[0]
        return render_template("index.html", results=round(prediction, 2))
    except Exception as e:
        return render_template("index.html", results=f"Error: {str(e)}")

# FastAPI setup for API integration
app = FastAPI()
app.mount("/", WSGIMiddleware(flask_app))

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/fastapi/predict")
def fastapi_predict(input_data: InsuranceInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_charges": round(prediction, 2)}
