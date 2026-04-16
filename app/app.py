import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
CHURN_THRESHOLD = 0.5

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = "Yes" if probability >= CHURN_THRESHOLD else "No"

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "threshold": CHURN_THRESHOLD,
        "risk_level": "high" if probability >= CHURN_THRESHOLD else "low",
    }
