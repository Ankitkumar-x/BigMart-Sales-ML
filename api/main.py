import os
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(
        0,
        SRC_PATH
    )


# ============================================================
# Import prediction pipeline
# ============================================================

from prediction_pipeline import (
    predict_sales,
    load_model_info
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="BigMart Sales Prediction API",
    description=(
        "Production API for BigMart Item Outlet Sales "
        "prediction using CatBoost."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# Input schema
# ============================================================

class SalesInput(BaseModel):

    Item_Identifier: str = Field(
        ...,
        min_length=1,
        description="Unique product identifier"
    )

    Item_Weight: float = Field(
        ...,
        gt=0,
        description="Product weight"
    )

    Item_Fat_Content: str = Field(
        ...,
        description="Product fat content"
    )

    Item_Visibility: float = Field(
        ...,
        ge=0,
        description="Product visibility"
    )

    Item_Type: str = Field(
        ...,
        description="Product category"
    )

    Item_MRP: float = Field(
        ...,
        gt=0,
        description="Maximum retail price"
    )

    Outlet_Identifier: str = Field(
        ...,
        min_length=1,
        description="Outlet identifier"
    )

    Outlet_Establishment_Year: int = Field(
        ...,
        ge=1980,
        le=2026,
        description="Outlet establishment year"
    )

    Outlet_Size: str = Field(
        ...,
        description="Outlet size"
    )

    Outlet_Location_Type: str = Field(
        ...,
        description="Outlet location tier"
    )

    Outlet_Type: str = Field(
        ...,
        description="Outlet type"
    )


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "BigMart Sales Prediction API"
    }


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "BigMart Sales Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ============================================================
# Model information
# ============================================================

@app.get("/model-info")
def model_info():

    info = load_model_info()

    return {
        "model": "CatBoostRegressor",
        "details": info
    }


# ============================================================
# Single prediction
# ============================================================

@app.post("/predict")
def predict(input_data: SalesInput):

    try:

        input_df = __import__(
            "pandas"
        ).DataFrame([
            input_data.model_dump()
        ])

        prediction = predict_sales(
            input_df
        )

        return {
            "prediction": round(
                float(prediction[0]),
                2
            ),
            "unit": "sales"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# Batch prediction
# ============================================================

@app.post("/predict-batch")
def predict_batch(
    inputs: List[SalesInput]
):

    if len(inputs) == 0:

        raise HTTPException(
            status_code=400,
            detail="Input list cannot be empty."
        )

    try:

        import pandas as pd

        input_df = pd.DataFrame(
            [
                item.model_dump()
                for item in inputs
            ]
        )

        predictions = predict_sales(
            input_df
        )

        results = []

        for index, prediction in enumerate(
            predictions
        ):

            results.append({
                "row": index,
                "prediction": round(
                    float(prediction),
                    2
                )
            })

        return {
            "count": len(results),
            "predictions": results
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(error)}"
        )