import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Paths / Imports ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import predict_on_test  

BUNDLE_PATH = PROJECT_ROOT / "models" / "car_price_bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)


USE_LOG_TARGET = False  

# ---------------- Feature engineering  ----------------
LUXURY = {"Audi", "BMW", "Mercedes", "Jaguar", "Porsche", "Lexus", "Volvo", "Land Rover"}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) car_age
    if "year" in df.columns:
        df["car_age"] = 2020 - df["year"]

    # 2) mileage_per_year
    if {"mileage", "year"}.issubset(df.columns):
        age = (2020 - df["year"]).replace(0, np.nan)
        df["mileage_per_year"] = (df["mileage"] / age).replace([np.inf, -np.inf], np.nan)

    # 3) is_high_end
    if "Brand" in df.columns:
        df["is_high_end"] = df["Brand"].isin(LUXURY).astype(int).fillna(0)

    # 4) engine_per_litre_efficiency
    if {"mpg", "engineSize"}.issubset(df.columns):
        denom = df["engineSize"].replace(0, np.nan)
        df["engine_per_litre_efficiency"] = (df["mpg"] / denom).replace([np.inf, -np.inf], np.nan)

    return df

def predict_price_from_dict(input_data: dict) -> float:
    df = pd.DataFrame([input_data])

    # Apply same feature engineering used in training
    df = add_features(df)


    df = df.drop(columns=["year"], errors="ignore")

    y_pred = predict_on_test(
        df,
        bundle["model"],
        scaler=bundle["scaler"],
        fill_values=bundle["fill_values"],
        selector=bundle["selector"],
        cat_modes=bundle["cat_modes"],
        freq_maps=bundle["freq_maps"],
        dummies_cols=bundle["dummies_cols"],
        feature_names=bundle["feature_names"],
        clip_info=bundle.get("clip_info", None),
        fe_freq_cols=("model",),  
        ohe_cols=("fuelType", "transmission", "Brand"),
        use_log_target=USE_LOG_TARGET,
    )

    return float(np.asarray(y_pred).ravel()[0])

# ---------------- UI ----------------
st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗")
st.title("Used Car Price Prediction")

col1, col2 = st.columns(2)

AVAILABLE_BRANDS = [
    "Audi", "BMW", "Ford", "Mercedes", "Volkswagen",
    "Toyota", "Skoda", "Hyundai", "Opel"
]

with col1:
    brand = st.selectbox("Brand", AVAILABLE_BRANDS)
    model = st.text_input("Model", "A3")
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
    fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    engine_size = st.number_input("Engine size (L)", min_value=0.5, max_value=6.0, value=1.6, step=0.1)

with col2:
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=120000, step=1000)
    tax = st.number_input("Annual tax (£)", min_value=0, max_value=1000, value=150)
    mpg = st.number_input("MPG", min_value=5.0, max_value=100.0, value=45.0, step=0.5)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
    previous_owners = st.number_input("Number of previous owners", min_value=0, max_value=10, value=1)

if st.button("Predict price"):
    input_data = {
        "Brand": brand,
        "model": model,
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "fuelType": fuel_type,
        "mpg": mpg,
        "engineSize": engine_size,
        "transmission": transmission,
        "previousOwners": previous_owners,
    }

    predicted_price = predict_price_from_dict(input_data)

    st.subheader("Estimated price")
    st.metric("Predicted price", f"{predicted_price:,.0f} £")
