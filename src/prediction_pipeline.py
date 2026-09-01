import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Paths
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(SRC_DIR, "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bigmart_catboost_final.cbm"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bigmart_feature_metadata.json"
)

MODEL_INFO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bigmart_model_metadata.json"
)


# ============================================================
# Load metadata
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


TARGET = metadata["target"]

CATEGORICAL_FEATURES = metadata[
    "categorical_features"
]


# ============================================================
# Load model
# ============================================================

model = CatBoostRegressor()

model.load_model(
    MODEL_PATH
)


# ============================================================
# Cleaning
# ============================================================

def clean_categorical_values(df):
    """
    Standardize categorical values.
    """

    df = df.copy()

    if "Item_Fat_Content" in df.columns:

        df["Item_Fat_Content"] = (
            df["Item_Fat_Content"]
            .replace({
                "LF": "Low Fat",
                "low fat": "Low Fat",
                "Low Fat": "Low Fat",
                "reg": "Regular",
                "Regular": "Regular"
            })
        )

    return df


# ============================================================
# Feature Engineering
# ============================================================

def final_feature_engineering(df):
    """
    Apply exactly the same feature engineering
    used during model training.
    """

    df = df.copy()

    df["Outlet_Age"] = (
        2026
        - df["Outlet_Establishment_Year"]
    )

    df["Item_ID_Category"] = (
        df["Item_Identifier"]
        .astype(str)
        .str[:2]
    )

    df["Item_Visibility_Zero"] = (
        df["Item_Visibility"] == 0
    ).astype(int)

    df["Item_Visibility_Log"] = (
        np.log1p(df["Item_Visibility"])
    )

    df["Item_MRP_Log"] = (
        np.log1p(df["Item_MRP"])
    )

    return df


# ============================================================
# Prepare Prediction Input
# ============================================================

def prepare_prediction_input(df):
    """
    Convert raw user data into the exact feature
    structure expected by CatBoost.
    """

    df = df.copy()

    df = clean_categorical_values(df)

    df = final_feature_engineering(df)

    for col in CATEGORICAL_FEATURES:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
            )

    return df


# ============================================================
# Prediction
# ============================================================

def predict_sales(df):
    """
    Predict sales for one or multiple rows.
    """

    prepared = prepare_prediction_input(df)

    predictions = model.predict(
        prepared
    )

    predictions = np.maximum(
        predictions,
        0
    )

    return np.asarray(predictions)


# ============================================================
# Model information
# ============================================================

def load_model_info():

    if not os.path.exists(
        MODEL_INFO_PATH
    ):
        return {}

    with open(
        MODEL_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)