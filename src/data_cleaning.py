"""
Data Cleaning Module
Customer Churn Prediction Project
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Convert TotalCharges to numeric
    2. Handle missing values
    3. Remove duplicates
    4. Drop customerID
    5. Encode binary target
    """
    df = df.copy()

    # 1. TotalCharges has whitespace strings → convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2. Missing values created above (11 rows) → fill with MonthlyCharges
    #    (new customers with tenure=0 have no TotalCharges yet)
    missing_mask = df["TotalCharges"].isna()
    df.loc[missing_mask, "TotalCharges"] = df.loc[missing_mask, "MonthlyCharges"]

    # 3. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"⚠️  Dropped {before - after} duplicate rows")

    # 4. Drop customerID — not predictive
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # 5. Encode target: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"✅ Cleaning complete. Final shape: {df.shape}")
    return df


def get_feature_types(df: pd.DataFrame):
    """Return lists of numerical and categorical feature columns."""
    target = "Churn"
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical = [c for c in numerical if c != target]
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    return numerical, categorical
