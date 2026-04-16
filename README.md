# Customer Churn Prediction

This project predicts whether a telecom customer is likely to leave the service (`Churn = Yes`). It is an end-to-end machine learning project that includes exploratory data analysis, preprocessing, model training, model persistence, and a FastAPI inference endpoint.

## Business goal

Customer churn prediction helps a company identify at-risk customers early and launch retention actions before they leave. In this project, recall for the churn class is especially important because missing a real churn customer can be costly for the business.

## Project structure

```text
churn-prediction/
├── app/
│   └── app.py
├── data/
│   └── data.csv
├── models/
│   └── model.pkl
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Dataset

- Source: Telco Customer Churn dataset
- Rows after cleaning: 7,032
- Target column: `Churn`

## EDA highlights

- The churn rate is about `26.6%`, so the classes are moderately imbalanced.
- `TotalCharges` required cleaning because it was loaded as text and contained invalid blank values.
- Customers with `Month-to-month` contracts showed higher churn risk.
- Customers using `Fiber optic` and `Electronic check` also showed higher churn risk.
- Customers who churned had lower average `tenure` and higher `MonthlyCharges`.

## Models compared

Three models were evaluated:

- Logistic Regression
- Random Forest
- Logistic Regression with `class_weight="balanced"`

### Test set results

| Model | Accuracy | Recall for `Yes` | F1 for `Yes` |
| --- | --- | --- | --- |
| Logistic Regression | 0.804 | 0.57 | 0.61 |
| Random Forest | 0.783 | 0.48 | 0.54 |
| Logistic Regression (`balanced`) | 0.726 | 0.80 | 0.61 |

## Final model choice

The final model is Logistic Regression with `class_weight="balanced"`.

Why this model:

- It significantly improved recall for churn customers.
- It reduced false negatives from `160` to `76`.
- It matched the business goal better than the higher-accuracy alternatives.

## Results summary

- Final model: Logistic Regression with balanced class weights
- Main business metric: recall for churn customers
- Final churn recall on the test set: `0.80`
- False negatives reduced from `160` to `76`
- API available through FastAPI with interactive Swagger docs

## Important features

Features associated with higher churn:

- `Contract_Month-to-month`
- `InternetService_Fiber optic`
- `PaymentMethod_Electronic check`
- `OnlineSecurity_No`
- `TechSupport_No`

Features associated with lower churn:

- `tenure`
- `Contract_Two year`
- `InternetService_DSL`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run EDA

Open the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Train the model

This script loads the dataset, cleans it, evaluates the candidate models, and saves the final pipeline:

```bash
python src/train.py
```

The trained pipeline will be stored in `models/model.pkl`.

## Run local prediction

```bash
python src/predict.py
```

Example output:

```text
Predicted churn: Yes
Churn probability: 0.8097
```

## Run the API

Start the FastAPI server:

```bash
python -m uvicorn app.app:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### Example API request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
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
  "TotalCharges": 29.85
}'
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8097,
  "threshold": 0.5,
  "risk_level": "high"
}
```

## Run with Docker

Build the image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

## Future improvements

- Add a Streamlit frontend for interactive customer scoring
- Tune the classification threshold for different business scenarios
- Add automated tests for the prediction pipeline and API
- Deploy the API to a cloud service such as Render or Railway
- Track experiments and metrics with an ML experiment tool

## Interview pitch

> I built an end-to-end churn prediction system. I started with EDA to understand the drivers of churn, cleaned and transformed the data with a reusable preprocessing pipeline, compared multiple classification models, and optimized for recall because missing a real churn customer is costly for the business. I then saved the model and exposed it through a FastAPI endpoint for real-time prediction.
