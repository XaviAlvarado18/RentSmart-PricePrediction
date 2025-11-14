"""
Data Preparation Pipeline for Rent Price Prediction - CuántoRento GT
Phase 3 - CRISP-DM: Data Preparation

This script performs:
- Schema validation
- Data Cleaning
- Imputation
- Feature Engineering
- Outlier winsorizing
- Encoding & Scaling
- Multicollinearity reduction
- Train/Validation/Test Split (temporal strategy)
- Output: processed datasets + scaler object
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# ============================================================
# EXPECTED SCHEMA VALIDATION
# ============================================================

EXPECTED_COLUMNS = {
    "listing_id", "zone", "property_type", "size_m2", "bedrooms", "bathrooms",
    "age_years", "floor", "has_elevator", "distance_to_business_km",
    "distance_to_transit_km", "view_quality", "noise_level", "has_security",
    "has_pool", "has_gym", "has_social", "furnished", "balcony", "garden",
    "parking_spaces", "pet_friendly", "rent_price_gtq", "price_per_m2",
    "age_bucket", "is_premium_zone"
}

def validate_schema(df):
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


# ============================================================
# HELPERS
# ============================================================

def winsorize(df, column, lower=0.01, upper=0.99):
    """Clip extreme outliers using quantiles."""
    low, high = df[column].quantile([lower, upper])
    df[column] = df[column].clip(low, high)
    return df


def normalize_booleans(df):
    bool_cols = [
        "has_security", "has_pool", "has_gym", "has_social",
        "furnished", "balcony", "garden", "pet_friendly", "has_elevator"
    ]
    for col in bool_cols:
        df[col] = df[col].astype(int)
    return df


def impute_missing(df):
    """Median for numeric, mode for categorical."""
    df = df.copy()
    numeric = df.select_dtypes(include=["int64", "float64"]).columns
    categorical = df.select_dtypes(include=["object", "category"]).columns

    df[numeric] = df[numeric].fillna(df[numeric].median())
    df[categorical] = df[categorical].fillna(df[categorical].mode().iloc[0])

    return df


# ============================================================
# 1. CLEANING
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, incorrect types, impossible values."""
    df = df.copy()

    # Schema validation
    validate_schema(df)

    # Standard cleaning
    df = df.drop_duplicates(subset="listing_id")

    # Remove negative numeric values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    # Winsorize outliers
    for col in ["rent_price_gtq", "size_m2", "distance_to_business_km"]:
        df = winsorize(df, col)

    # Remove impossible records
    df = df[df["size_m2"].between(20, 500)]
    df = df[df["rent_price_gtq"] > 0]

    # Imputation (in case synthetic data changes)
    df = impute_missing(df)

    # Normalize booleans
    df = normalize_booleans(df)

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price per bedroom
    df["price_per_bedroom"] = df["rent_price_gtq"] / df["bedrooms"]

    # Premium score
    df["premium_features_score"] = (
        df["has_security"] * 0.3 +
        df["has_pool"] * 0.2 +
        df["has_gym"] * 0.2 +
        df["has_social"] * 0.1 +
        df["parking_spaces"] * 0.2
    )

    # Ratio of distances
    df["distance_ratio"] = (
        df["distance_to_business_km"] /
        df["distance_to_transit_km"]
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Amenities count
    amenity_cols = [
        "has_security", "has_pool", "has_gym", "has_social",
        "furnished", "balcony", "garden", "pet_friendly"
    ]
    df["amenities_total"] = df[amenity_cols].sum(axis=1)

    return df


# ============================================================
# 3. FEATURE REDUCTION / ENCODING & SCALING
# ============================================================

def remove_high_corr_features(X, threshold=0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X = X.drop(columns=to_drop)
    return X, to_drop


def encode_and_scale(df: pd.DataFrame):
    df = df.copy()

    # Encoding
    categorical_cols = ["zone", "property_type", "age_bucket"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate target
    y = df["rent_price_gtq"]
    X = df.drop(["rent_price_gtq", "listing_id"], axis=1)

    # Remove multicollinearity
    X, dropped_cols = remove_high_corr_features(X)
    print("Dropped highly correlated features:", dropped_cols)

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, scaler


# ============================================================
# 4. SPLIT
# ============================================================

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def prepare_data(df: pd.DataFrame):
    df = df.sort_values("listing_id")

    df = clean_data(df)
    df = add_features(df)
    X, y, scaler = encode_and_scale(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save artifacts for reproducibility
    X.to_csv("../data/processed/feature_columns.csv", index=False)
    joblib.dump(scaler, "../data/processed/scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ============================================================
# EXECUTION EXAMPLE
# ============================================================

if __name__ == "__main__":
    sample = pd.read_csv("../data/raw/rent_guatemala.csv")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(sample)

    print("Shapes:")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)
