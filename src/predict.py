import pickle
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def predict_customer(customer_data: dict) -> tuple[str, float]:
    model = load_model()
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)[0]
    probability = float(model.predict_proba(customer_df)[0][1])
    return prediction, probability


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    prediction, probability = predict_customer(sample_customer)
    print("Predicted churn:", prediction)
    print("Churn probability:", probability)
