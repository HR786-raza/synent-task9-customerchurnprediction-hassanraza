"""
Preprocessing & Feature Engineering Module
Customer Churn Prediction Project
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add meaningful derived features."""
    df = df.copy()

    # Tenure group — buckets customers into lifecycle stages
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"],
        include_lowest=True,
    )

    # Service count — how many add-on services the customer uses
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # Has any streaming — entertainment stickiness
    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # Average monthly spend per tenure month
    df["AvgMonthlySpend"] = df.apply(
        lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
        axis=1,
    )

    return df


def build_preprocessor(numerical_cols, categorical_cols):
    """Build a ColumnTransformer with scaling + one-hot encoding."""
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Full data preparation:
    - Feature engineering
    - Train/test split (stratified)
    - Return X_train, X_test, y_train, y_test, preprocessor, feature lists
    """
    df = engineer_features(df)

    target = "Churn"
    y = df[target]
    X = df.drop(columns=[target])

    # Define feature sets after engineering
    numerical_cols = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "ServiceCount", "AvgMonthlySpend",
    ]
    categorical_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "TenureGroup", "HasStreaming",
    ]

    # Keep only columns that exist in X
    numerical_cols = [c for c in numerical_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    return X_train, X_test, y_train, y_test, preprocessor, numerical_cols, categorical_cols
