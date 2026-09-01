import os
import sys
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


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

# ============================================================
# FastAPI configuration
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
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
    prepare_prediction_input,
    model,
    CATEGORICAL_FEATURES,
    load_model_info
)


# ============================================================
# FastAPI prediction helper
# ============================================================

def predict_via_api(input_data):
    """
    Send a single prediction request to FastAPI.
    Returns the API JSON response.
    Raises a clear error when the API is unavailable.
    """

    payload = input_data.iloc[0].to_dict()

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "FastAPI is not reachable. "
            "Please start the API server."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "FastAPI request timed out."
        )

    except requests.exceptions.HTTPError:

        try:
            detail = response.json().get(
                "detail",
                "API returned an error."
            )
        except Exception:
            detail = "API returned an error."

        raise RuntimeError(
            f"FastAPI error: {detail}"
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"API request failed: {error}"
        )
    
# ============================================================
# FastAPI batch prediction helper
# ============================================================

def predict_batch_via_api(input_data):
    """
    Send batch prediction requests to FastAPI.
    Returns a list of predictions.
    """

    payload = input_data.to_json(
        orient="records"
    )

    import json

    payload = json.loads(
        payload
    )

    try:

        response = requests.post(
            f"{API_URL}/predict-batch",
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        return [
            item["prediction"]
            for item in result["predictions"]
        ]

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "FastAPI is not reachable. "
            "Please start the API server."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Batch prediction request timed out."
        )

    except requests.exceptions.HTTPError:

        try:
            detail = response.json().get(
                "detail",
                "API returned an error."
            )
        except Exception:
            detail = "API returned an error."

        raise RuntimeError(
            f"FastAPI error: {detail}"
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Batch API request failed: {error}"
        )
    
# ============================================================
# Optional SHAP import
# ============================================================

try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 6px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 17px;
        margin-bottom: 20px;
        opacity: 0.75;
    }

    /* Prediction card */
    .prediction-card {
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 18px;
        margin-bottom: 18px;
        text-align: center;
    }

    .metric-title {
        font-size: 16px;
        opacity: 0.75;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 42px;
        font-weight: 700;
        line-height: 1.1;
    }

    /* Section titles */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    /* Info cards */
    .info-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        margin-bottom: 10px;
    }

    /* Small explanation cards */
    .explanation-card {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        margin-bottom: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.6;
        padding: 25px 0 10px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🛒 BigMart Sales Predictor
    </div>

    <div class="subtitle">
        AI-powered retail sales prediction using a trained
        CatBoost regression model.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# Model information
# ============================================================

model_info = load_model_info()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write(
        "**Model:** CatBoost Regressor"
    )

    if model_info:

        st.metric(
            "Validation R²",
            f"{model_info.get('validation_r2', 0) * 100:.2f}%"
        )

        st.write(
            f"**MAE:** ₹{model_info.get('validation_mae', 0):,.2f}"
        )

        st.write(
            f"**RMSE:** ₹{model_info.get('validation_rmse', 0):,.2f}"
        )

    st.divider()

    st.caption(
        "Model trained on the BigMart sales dataset."
    )

    # ============================================================
    # API Status
    # ============================================================

    st.divider()

    st.subheader(
        "🔌 API Status"
    )
    
    try:
    
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
    
        if health_response.ok:
    
            st.success(
                "🟢 API Online"
            )
    
            st.caption(
                "Backend is ready to serve predictions."
            )
    
        else:
    
            st.error(
                "🔴 API Unavailable"
            )
    
    except requests.RequestException:
    
        st.error(
            "🔴 API Offline"
        )
    
        st.caption(
            "Start FastAPI to enable predictions."
        )
# ============================================================
# SHAP Helper
# ============================================================

@st.cache_resource
def get_shap_explainer():
    """
    Create and cache the SHAP TreeExplainer.
    This prevents rebuilding the explainer on every interaction.
    """

    if not SHAP_AVAILABLE:
        return None

    return shap.TreeExplainer(model)
    

# ============================================================
# Tabs
# ============================================================

tab_single, tab_batch, tab_info = st.tabs(
    [
        "🔮 Single Prediction",
        "📂 Batch Prediction",
        "📊 Model Information"
    ]
)
st.divider()

st.subheader(
    "🏗️ System Architecture"
)

st.code(
    """
Streamlit UI
     ↓
FastAPI REST API
     ↓
Prediction Pipeline
     ↓
CatBoost Model
    """,
    language="text"
)

# ============================================================
# TAB 1 — SINGLE PREDICTION
# ============================================================

with tab_single:

    st.markdown(
        '<div class="section-title">📦 Product Information</div>',
        unsafe_allow_html=True
    )

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            item_identifier = st.text_input(
                "Item Identifier",
                value="FDA15"
            )

            item_weight = st.number_input(
                "Item Weight",
                min_value=0.0,
                value=10.0,
                step=0.1
            )

            item_fat_content = st.selectbox(
                "Item Fat Content",
                [
                    "Low Fat",
                    "Regular"
                ]
            )

            item_visibility = st.number_input(
                "Item Visibility",
                min_value=0.0,
                value=0.05,
                step=0.001,
                format="%.4f"
            )

        with col2:

            item_type = st.selectbox(
                "Item Type",
                [
                    "Baking Goods",
                    "Breads",
                    "Breakfast",
                    "Canned",
                    "Dairy",
                    "Frozen Foods",
                    "Fruits and Vegetables",
                    "Hard Drinks",
                    "Health and Hygiene",
                    "Household",
                    "Meat",
                    "Others",
                    "Seafood",
                    "Snack Foods",
                    "Soft Drinks",
                    "Starchy Foods"
                ]
            )

            item_mrp = st.number_input(
                "Item MRP",
                min_value=0.0,
                value=150.0,
                step=1.0
            )

        st.markdown(
            '<div class="section-title">🏪 Outlet Information</div>',
            unsafe_allow_html=True
        )

        col3, col4 = st.columns(2)

        with col3:

            outlet_identifier = st.selectbox(
                "Outlet Identifier",
                [
                    "OUT010",
                    "OUT013",
                    "OUT017",
                    "OUT018",
                    "OUT019",
                    "OUT027",
                    "OUT035",
                    "OUT045",
                    "OUT046",
                    "OUT049"
                ]
            )

            outlet_size = st.selectbox(
                "Outlet Size",
                [
                    "Small",
                    "Medium",
                    "High"
                ]
            )

            outlet_location = st.selectbox(
                "Outlet Location Type",
                [
                    "Tier 1",
                    "Tier 2",
                    "Tier 3"
                ]
            )

        with col4:

            outlet_type = st.selectbox(
                "Outlet Type",
                [
                    "Grocery Store",
                    "Supermarket Type1",
                    "Supermarket Type2",
                    "Supermarket Type3"
                ]
            )

            establishment_year = st.number_input(
                "Outlet Establishment Year",
                min_value=1980,
                max_value=2026,
                value=2000,
                step=1
            )

        submitted = st.form_submit_button(
            "🔮 Predict Sales",
            use_container_width=True
        )


    # ========================================================
    # Prediction
    # ========================================================

    if submitted:

        # -----------------------------
        # Input validation
        # -----------------------------

        errors = []

        if not item_identifier.strip():

            errors.append(
                "Item Identifier cannot be empty."
            )

        if item_weight <= 0:

            errors.append(
                "Item Weight must be greater than 0."
            )

        if item_visibility < 0:

            errors.append(
                "Item Visibility cannot be negative."
            )

        if item_mrp <= 0:

            errors.append(
                "Item MRP must be greater than 0."
            )

        if (
            establishment_year < 1980
            or establishment_year > 2026
        ):

            errors.append(
                "Please enter a valid outlet establishment year."
            )


        if errors:

            for error in errors:

                st.error(error)

        else:

            input_data = pd.DataFrame({

                "Item_Identifier": [
                    item_identifier.strip()
                ],

                "Item_Weight": [
                    item_weight
                ],

                "Item_Fat_Content": [
                    item_fat_content
                ],

                "Item_Visibility": [
                    item_visibility
                ],

                "Item_Type": [
                    item_type
                ],

                "Item_MRP": [
                    item_mrp
                ],

                "Outlet_Identifier": [
                    outlet_identifier
                ],

                "Outlet_Establishment_Year": [
                    establishment_year
                ],

                "Outlet_Size": [
                    outlet_size
                ],

                "Outlet_Location_Type": [
                    outlet_location
                ],

                "Outlet_Type": [
                    outlet_type
                ]
            })


            try:

                api_result = predict_via_api(
                    input_data
                )
                
                predicted_sales = float(
                    api_result["prediction"]
                )

                st.success(
                    "Prediction generated successfully."
                )

                # -----------------------------
                # Prediction card
                # -----------------------------

                st.markdown(
                    f"""
                    <div class="prediction-card">

                    <div class="metric-title">
                    Predicted Item Outlet Sales
                    </div>

                    <div class="metric-value">
                    ₹ {predicted_sales:,.2f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ==================================================
                # Prediction Summary
                # ==================================================
                
                st.subheader(
                    "🎯 Prediction Summary"
                )
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                
                    st.metric(
                        "Predicted Sales",
                        f"₹ {predicted_sales:,.2f}"
                    )
                
                with summary_col2:
                
                    st.metric(
                        "Model R²",
                        f"{model_info.get('validation_r2', 0) * 100:.2f}%"
                    )
                
                with summary_col3:
                
                    st.metric(
                        "Model",
                        "CatBoost"
                    )
                
                st.caption(
                    "This prediction is an estimate generated by the trained "
                    "machine-learning model and treated as a "
                    "decision-support signal."
                )


                # ==================================================
                # SHAP Explanation
                # ==================================================

                if SHAP_AVAILABLE:

                    st.subheader(
                        "🔍 Why did the model make this prediction?"
                    )

                    try:

                        prepared_input = (
                            prepare_prediction_input(
                                input_data
                            )
                        )

                        explainer = get_shap_explainer()

                        shap_result = explainer(
                            prepared_input
                        )

                        st.write(
                            "The chart below shows how each feature "
                            "moved the prediction away from the model's "
                            "baseline prediction."
                        )

                        # ------------------------------------------
                        # SHAP Waterfall Chart
                        # ------------------------------------------

                        shap.plots.waterfall(
                            shap_result[0],
                            max_display=8,
                            show=False
                        )

                        st.pyplot(
                            plt.gcf(),
                            clear_figure=True
                        )

                        plt.close("all")

                        # ------------------------------------------
                        # Feature Contributions
                        # ------------------------------------------

                        shap_values_row = (
                            shap_result[0].values
                        )

                        feature_names = (
                            prepared_input.columns.tolist()
                        )

                        contribution_df = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP_Value": shap_values_row
                        })

                        contribution_df["Impact"] = np.where(
                            contribution_df["SHAP_Value"] >= 0,
                            "Increases prediction",
                            "Decreases prediction"
                        )

                        contribution_df["Absolute_Impact"] = (
                            contribution_df["SHAP_Value"].abs()
                        )

                        contribution_df = (
                            contribution_df
                            .sort_values(
                                "Absolute_Impact",
                                ascending=False
                            )
                            .reset_index(drop=True)
                        )

                        contribution_df["SHAP_Value"] = (
                            contribution_df["SHAP_Value"].round(2)
                        )

                        st.subheader(
                            "Top Feature Contributions"
                        )

                        st.dataframe(
                            contribution_df[
                                [
                                    "Feature",
                                    "SHAP_Value",
                                    "Impact"
                                ]
                            ].head(10),
                            use_container_width=True,
                            hide_index=True
                        )

                        # ------------------------------------------
                        # Positive contributors
                        # ------------------------------------------

                        positive_contributors = (
                            contribution_df[
                                contribution_df["SHAP_Value"] > 0
                            ]
                            .head(3)
                        )

                        if not positive_contributors.empty:

                            st.write(
                                "### ⬆️ Factors increasing prediction"
                            )

                            for _, row in (
                                positive_contributors.iterrows()
                            ):

                                st.markdown(
                                    f"""
                                    **{row['Feature']}**
                                    → +{row['SHAP_Value']:.2f}
                                    """
                                )

                        # ------------------------------------------
                        # Negative contributors
                        # ------------------------------------------

                        negative_contributors = (
                            contribution_df[
                                contribution_df["SHAP_Value"] < 0
                            ]
                            .head(3)
                        )

                        if not negative_contributors.empty:

                            st.write(
                                "### ⬇️ Factors decreasing prediction"
                            )

                            for _, row in (
                                negative_contributors.iterrows()
                            ):

                                st.markdown(
                                    f"""
                                    **{row['Feature']}**
                                    → {row['SHAP_Value']:.2f}
                                    """
                                )

                    except Exception as shap_error:

                        st.warning(
                            "Prediction succeeded, but the SHAP "
                            "explanation could not be generated."
                        )

                        st.error(
                            str(shap_error)
                        )

                else:

                    st.info(
                        "SHAP is not installed. Run "
                        "`pip install shap` to enable explanations."
                    )


            except Exception as prediction_error:

                st.error(
                    "⚠️ Prediction failed."
                )

                st.exception(
                    str(prediction_error)
                )



# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tab_batch:

    st.subheader(
        "📂 Batch CSV Prediction"
    )

    st.write(
        "Upload a CSV containing BigMart product and outlet features."
    )

    st.info(
        """
        Required columns:

        Item_Identifier, Item_Weight, Item_Fat_Content,
        Item_Visibility, Item_Type, Item_MRP,
        Outlet_Identifier, Outlet_Establishment_Year,
        Outlet_Size, Outlet_Location_Type, Outlet_Type
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # ---------------------------------------------
            # Read uploaded CSV
            # ---------------------------------------------

            batch_df = pd.read_csv(
                uploaded_file
            )

            # Remove accidental index columns
            batch_df = batch_df.loc[
                :,
                ~batch_df.columns.str.startswith("Unnamed:")
            ]

            # ---------------------------------------------
            # Preview
            # ---------------------------------------------

            st.write(
                "Uploaded data preview:"
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            # ---------------------------------------------
            # Required columns
            # ---------------------------------------------

            required_columns = [
                "Item_Identifier",
                "Item_Weight",
                "Item_Fat_Content",
                "Item_Visibility",
                "Item_Type",
                "Item_MRP",
                "Outlet_Identifier",
                "Outlet_Establishment_Year",
                "Outlet_Size",
                "Outlet_Location_Type",
                "Outlet_Type"
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in batch_df.columns
            ]

            # ---------------------------------------------
            # Missing-column validation
            # ---------------------------------------------

            if missing_columns:

                st.error(
                    "Missing required columns:"
                )

                st.write(
                    missing_columns
                )

            elif batch_df.empty:

                st.warning(
                    "The uploaded CSV is empty."
                )

            else:

                # -----------------------------------------
                # Data validation
                # -----------------------------------------

                validation_errors = []

                if (
                    batch_df["Item_Weight"]
                    .notna()
                    .any()
                    and (
                        batch_df["Item_Weight"].dropna() <= 0
                    ).any()
                ):
                    validation_errors.append(
                        "Item_Weight must be greater than 0."
                    )

                if (
                    batch_df["Item_Visibility"]
                    .notna()
                    .any()
                    and (
                        batch_df["Item_Visibility"].dropna() < 0
                    ).any()
                ):
                    validation_errors.append(
                        "Item_Visibility cannot be negative."
                    )

                if (
                    batch_df["Item_MRP"]
                    .notna()
                    .any()
                    and (
                        batch_df["Item_MRP"].dropna() <= 0
                    ).any()
                ):
                    validation_errors.append(
                        "Item_MRP must be greater than 0."
                    )

                if (
                    batch_df["Outlet_Establishment_Year"]
                    .notna()
                    .any()
                ):

                    invalid_years = batch_df[
                        "Outlet_Establishment_Year"
                    ].dropna()

                    if (
                        (invalid_years < 1980)
                        |
                        (invalid_years > 2026)
                    ).any():

                        validation_errors.append(
                            "Outlet_Establishment_Year contains "
                            "invalid values."
                        )

                # -----------------------------------------
                # Show validation errors
                # -----------------------------------------

                if validation_errors:

                    for error in validation_errors:

                        st.error(
                            error
                        )

                else:

                    st.success(
                        f"{len(batch_df):,} rows ready for prediction."
                    )

                    # -------------------------------------
                    # Generate predictions
                    # -------------------------------------

                    if st.button(
                        "🚀 Generate Batch Predictions",
                        use_container_width=True
                    ):

                        try:

                            predictions = predict_batch_via_api(
                                batch_df[
                                    required_columns
                                ]
                            )

                            result_df = (
                                batch_df.copy()
                            )

                            result_df[
                                "Predicted_Item_Outlet_Sales"
                            ] = np.round(
                                predictions,
                                2
                            )

                            # ---------------------------------
                            # Summary calculations
                            # ---------------------------------

                            total_predicted_sales = (
                                result_df[
                                    "Predicted_Item_Outlet_Sales"
                                ].sum()
                            )

                            average_predicted_sales = (
                                result_df[
                                    "Predicted_Item_Outlet_Sales"
                                ].mean()
                            )

                            max_predicted_sales = (
                                result_df[
                                    "Predicted_Item_Outlet_Sales"
                                ].max()
                            )

                            # ---------------------------------
                            # Success message
                            # ---------------------------------

                            st.success(
                                "Batch prediction completed."
                            )

                            # ---------------------------------
                            # Summary cards
                            # ---------------------------------

                            st.subheader(
                                "📊 Batch Prediction Summary"
                            )

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                            
                                st.metric(
                                    "Rows",
                                    f"{len(result_df):,}"
                                )
                            
                            with col2:
                            
                                st.metric(
                                    "Total Predicted Sales",
                                    f"₹ {total_predicted_sales:,.2f}"
                                )
                            
                            with col3:
                            
                                st.metric(
                                    "Average Predicted Sales",
                                    f"₹ {average_predicted_sales:,.2f}"
                                )
                            
                            with col4:
                            
                                st.metric(
                                    "Highest Prediction",
                                    f"₹ {max_predicted_sales:,.2f}"
                                )

                            # ---------------------------------
                            # Prediction results
                            # ---------------------------------

                            st.subheader(
                                "Prediction Results"
                            )

                            st.dataframe(
                                result_df.head(20),
                                use_container_width=True,
                                hide_index=True
                            )

                            # ---------------------------------
                            # Download
                            # ---------------------------------

                            csv_data = (
                                result_df
                                .to_csv(index=False)
                                .encode("utf-8")
                            )

                            st.download_button(
                                label="⬇️ Download Predictions CSV",
                                data=csv_data,
                                file_name=(
                                    "BigMart_Predictions.csv"
                                ),
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as batch_error:

                            st.error(
                                "⚠️ Batch prediction failed."
                            )

                            st.exception(
                                str(batch_error)
                            )

        except Exception as upload_error:

            st.error(
                "Could not read the uploaded CSV."
            )

            st.exception(
                upload_error
            )


# ============================================================
# TAB 3 — MODEL INFORMATION
# ============================================================

with tab_info:

    st.subheader(
        "📊 Model Information"
    )

    st.write(
        "The production model is a CatBoost regression model."
    )

    if model_info:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Validation R²",
                f"{model_info.get('validation_r2', 0) * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Validation MAE",
                f"₹{model_info.get('validation_mae', 0):,.2f}"
            )

        with col3:

            st.metric(
                "Validation RMSE",
                f"₹{model_info.get('validation_rmse', 0):,.2f}"
            )

        st.divider()

        st.write(
            "**Algorithm:** CatBoostRegressor"
        )

        st.write(
            f"**Iterations:** "
            f"{model_info.get('iterations', 'N/A')}"
        )

        st.write(
            f"**Depth:** "
            f"{model_info.get('depth', 'N/A')}"
        )

        st.write(
            f"**Learning Rate:** "
            f"{model_info.get('learning_rate', 'N/A')}"
        )

        st.write(
            f"**Number of Features:** "
            f"{model_info.get('feature_count', 'N/A')}"
        )

        st.write(
            "**Primary objective:** "
            "Predict Item_Outlet_Sales."
        )

    else:

        st.warning(
            "Model metadata is not available."
        )

st.markdown(
    """
    <div class="footer">
        BigMart Sales Prediction • CatBoost • Explainable AI • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)        