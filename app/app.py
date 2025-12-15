import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# App: Used Car Price Prediction (Streamlit)

# This file provides a simple UI where a user can enter car attributes
# (brand, model, year, mileage, etc.) and obtain a predicted price.

# Project path setup

# When running Streamlit, the working directory might be different depending on
# how the app is launched. To keep imports stable, we add the project root to
# sys.path so that `from src...` works reliably.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.modeling import predict_on_test



# Load trained bundle

# The bundle contains the trained estimator and all preprocessing artifacts
# learned during training (scaler, feature selector, fill values, dummy columns,
# feature_names, etc.). This allows prediction without re-running the notebook.

BUNDLE_PATH = PROJECT_ROOT / "models" / "car_price_bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)


USE_LOG_TARGET = True  

# ---------------- Feature engineering  ----------------

def norm_text(s: pd.Series) -> pd.Series:

  ''' Function to normalize model categories:
  - manages differring representations of characters
  - removes the spaces before and after
  - converts to lowercase
  - convert multiple spaces to one'''

  return (s.astype('string')
             .str.normalize('NFKC') # to manage different representations of certain characters
             .str.strip() # remove spaces before and after
             .str.lower() # convert everything to lower case
             .str.replace(r'\s+', ' ', regex=True)) # to convert multiple spaces to only one

def clean_model(model: pd.Series) -> pd.Series:

  ''' Works in sync with norm_text function, that performs the initial normalization.
   After ensuring the values are all written within the same patters (norm_text function),
   we have to correct the lack of 's' in the end of some words, and the usage of '-' within the
   names of some models. We also included specific corrections for some models. Aditionally, we ensure only a-z,
   0-9 and +- are allowed, and replace empty categories (withe spaces) by Nan'''
  
  m = norm_text(model)

    # 1) Semantic normalization - fix common truncations/typos, so classes are uniform
  m = (m
         # BMW: serie -> series
        .str.replace(r'\bserie\b', 'series', regex=True)
         # Mercedes: clas/claaass -> class
        .str.replace(r'\bclas+\b', 'class', regex=True)
         # Unificar t-roc / t roc
        .str.replace(r'\bt\s*[- ]\s*roc\b', 't-roc', regex=True)
        .str.replace(r'\bt\s*[- ]\s*cross\b', 't-cross', regex=True)
        # Opel Combo: lif -> life
        .str.replace(r'\bcombo lif\b', 'combo life', regex=True)
        # Caddy Maxi: lif -> life
        .str.replace(r'\bcaddy maxi lif\b', 'caddy maxi life', regex=True)
        # Ford Edge: edg -> edge
        .str.replace(r'\bedg\b', 'edge', regex=True)
        # many other imcomplete models
        .str.replace(r'\bcors\b', 'corsa', regex=True)
        .str.replace(r'\bmokk\b', 'mokka', regex=True)
        .str.replace(r'\btucso\b', 'tucson', regex=True)
        .str.replace(r'\btigua\b', 'tiguan', regex=True)
        .str.replace(r'\bhilu\b', 'hilux', regex=True)
        .str.replace(r'\bvers\b', 'verso', regex=True)
        .str.replace(r'\byari\b', 'yaris', regex=True)
        .str.replace(r'\byet\b', 'yeti', regex=True)
        .str.replace(r'\btourneo custo\b', 'tourneo custom', regex=True)
        .str.replace(r'\brav\b', 'rav4', regex=True)
        .str.replace(r'\bs ma\b', 's-max', regex=True)
        .str.replace(r'\bscirocc\b', 'scirocco', regex=True)
        .str.replace(r'\btouare\b', 'touareg', regex=True)
        .str.replace(r'\bcoroll\b', 'corolla', regex=True)
        .str.replace(r'\bamaro\b', 'amarok', regex=True)
        .str.replace(r'\bcruise\b', 'cruiser', regex=True)
        .str.replace(r'\btoure\b', 'tourer', regex=True)
        .str.replace(r'\boutdoo\b', 'outdoor', regex=True)
        .str.replace(r'\btoura\b', 'touran', regex=True)
        .str.replace(r'\ballspac\b', 'allspace', regex=True)
        .str.replace(r'\becospor\b', 'ecosport', regex=True)
        .str.replace(r'\bzafir\b', 'zafira', regex=True)
        .str.replace(r'\bkon\b', 'kona', regex=True)
        .str.replace(r'\bmeriv\b', 'meriva', regex=True)
        .str.replace(r'\bt-cros\b', 't-cross', regex=True)
        .str.replace(r'\bt-ro\b', 't-roc', regex=True)
        .str.replace(r'\bs-ma\b', 's-max', regex=True)
        .str.replace(r'\binsigni\b', 'insignia', regex=True)
        .str.replace(r'\bioni\b', 'ioniq', regex=True)
        .str.replace(r'\bada\b', 'adam', regex=True)
        .str.replace(r'\barteo\b', 'arteon', regex=True)
        .str.replace(r'\bastr\b', 'astra', regex=True)
        .str.replace(r'\bauri\b', 'auris', regex=True)
        .str.replace(r'\bayg\b', 'aygo', regex=True)
        .str.replace(r'\bb-ma\b', 'b-max', regex=True)
        .str.replace(r'\bbeetl\b', 'beetle', regex=True)
        .str.replace(r'\bc h\b', 'c-hr', regex=True)
        .str.replace(r'\bc-h\b', 'c-hr', regex=True)
        .str.replace(r'\bc-ma\b', 'c-max', regex=True)
        .str.replace(r'\bc ma\b', 'c-max', regex=True)
        .str.replace(r'\bkami\b', 'kamiq', regex=True)
        .str.replace(r'\bkaro\b', 'karoq', regex=True)
        .str.replace(r'\bkodia\b', 'kodiaq', regex=True)
        .str.replace(r'\bkug\b', 'kuga', regex=True)
        .str.replace(r'\bmonde\b', 'mondeo', regex=True)
        .str.replace(r'\boctavi\b', 'octavia', regex=True)
        .str.replace(r'\bpassa\b', 'passat', regex=True)
        .str.replace(r'\bpol\b', 'polo', regex=True)
        .str.replace(r'\brapi\b', 'rapid', regex=True)
        .str.replace(r'\broomste\b', 'roomster', regex=True)
        .str.replace(r'\bs-ma\b', 's-max', regex=True)
        .str.replace(r'\bsanta f\b', 'santa fe', regex=True)
        .str.replace(r'\bscal\b', 'scala', regex=True)
        .str.replace(r'\bcaravell\b', 'caravelle', regex=True)
        .str.replace(r'\bcitig\b', 'citigo', regex=True)
        .str.replace(r'\bfabi\b', 'fabia', regex=True)
        .str.replace(r'\bfiest\b', 'fiesta', regex=True)
        .str.replace(r'\bfocu\b', 'focus', regex=True)
        .str.replace(r'\bgalax\b', 'galaxy', regex=True)
        .str.replace(r'\bgol\b', 'golf', regex=True)
        .str.replace(r'\bgrand c-ma\b', 'grand c-max', regex=True)
        .str.replace(r'\bgrand tourneo connec\b', 'grand tourneo connect', regex=True)
        .str.replace(r'\bsuper\b', 'superb', regex=True)
        .str.replace(r'\bviv\b', 'vivaro', regex=True)
        .str.replace(r'\bviva\b', 'vivaro', regex=True)
    )

    # 2) only maintaining specific simbols: a-z; 0-9 and + -
  m = (m
         .str.replace(r'[^a-z0-9\+\- ]', ' ', regex=True)
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip())

    # Treat empty and very short strings as missing
  m = m.replace('', pd.NA)
  m = m.mask(m.str.len() == 1, pd.NA)
  return m

# Luxury brands flag used during training feature engineering
LUXURY = {"Audi", "BMW", "Mercedes", "Jaguar", "Porsche", "Lexus", "Volvo", "Land Rover"}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Deterministic feature engineering used during training.
    
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


# Prediction wrapper

def predict_price_from_dict(input_data: dict) -> float:
    """
    Convert a raw user input dictionary into a 1-row DataFrame, apply the same
    cleaning/feature engineering as training, then call predict_on_test using
    the preprocessing artifacts stored in the bundle.
    """
    df = pd.DataFrame([input_data])

    # Normalize text inputs to match training preprocessing
    for col in ["Brand", "fuelType", "transmission"]:
        if col in df.columns:
            df[col] = norm_text(df[col])

    if "model" in df.columns:
        df["model"] = clean_model(df["model"])
    
    # Apply training-consistent engineered features

    df = add_features(df)

    # Drop raw 'year'
    df = df.drop(columns=["year"], errors="ignore")
    
    # Predict using the trained pipeline objects stored in the bundle
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

# Streamlit UI
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
    # Build the input row using the exact column names used in the model pipeline
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




