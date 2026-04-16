from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERICAL_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw churn dataset from CSV."""
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and drop rows with invalid values."""
    cleaned_df = df.copy()
    cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["TotalCharges"])
    return cleaned_df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target column."""
    model_df = df.drop(columns=DROP_COLUMNS)
    features = model_df.drop(columns=[TARGET_COLUMN])
    target = model_df[TARGET_COLUMN]
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for numerical and categorical data."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
