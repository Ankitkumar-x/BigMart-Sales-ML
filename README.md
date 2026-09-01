# 🛒 BigMart Sales Prediction

An end-to-end machine learning system for predicting `Item_Outlet_Sales` using advanced regression models, explainable AI, a Streamlit frontend, FastAPI backend, Docker, and cloud deployment.

## 🌐 Live Application

**Streamlit Application:**  
https://bigmart-sales-ml.onrender.com

**FastAPI Backend:**  
https://sales-v8m6.onrender.com

**API Documentation:**  
https://sales-v8m6.onrender.com/docs

---

# 📌 Project Overview

The objective of this project is to predict the sales of a product at a particular retail outlet using product-level and outlet-level characteristics.

The project was developed as a complete production-oriented machine learning system rather than only a notebook experiment.

The final architecture includes:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Multiple regression models
- Cross-validation
- Hyperparameter optimization
- Model comparison
- Ensemble and stacking experiments
- CatBoost production model
- SHAP explainability
- Streamlit UI
- FastAPI REST API
- Docker
- Docker Compose
- GitHub
- Render cloud deployment

---

# 🎯 Business Problem

Retail businesses need accurate estimates of product sales to support inventory planning, store-level decisions, and business analysis.

The model predicts:

`Item_Outlet_Sales`

using information such as:

- Product identifier
- Product weight
- Fat content
- Product visibility
- Product category
- Product MRP
- Outlet identifier
- Outlet establishment year
- Outlet size
- Outlet location type
- Outlet type

---

# 📊 Dataset

The project uses the BigMart sales dataset containing product and outlet information.

The target variable is:


Item_Outlet_Sales

# 🔎 Exploratory Data Analysis

EDA was performed to understand:

- Dataset structure
- Missing values
- Duplicate records
- Unique values
- Numerical distributions
- Categorical distributions
- Outliers
- Target distribution
- Feature correlations
- Feature-target relationships

The goal of EDA was to identify data-quality issues and discover useful patterns before model development.

## 🛠️ Feature Engineering

Several additional features were created to improve the representation of the underlying business problem.

Examples include:

### Outlet Age
```
Outlet_Age = Current Year - Outlet_Establishment_Year
```

### Item Identifier Category
The first two characters of the product identifier were extracted as a higher-level product category.

### Zero Visibility Indicator
`Item_Visibility_Zero` identifies products with zero recorded visibility.

### Log-transformed Visibility
`Item_Visibility_Log` was created using:
```python
np.log1p(Item_Visibility)
```

### Log-transformed MRP
`Item_MRP_Log` was created using:
```python
np.log1p(Item_MRP)
```

## 🤖 Models Evaluated

Multiple regression algorithms were evaluated under the same validation framework.

| Model | Validation R² |
|-------|---------------|
| CatBoost | 0.6174 |
| XGBoost | 0.6115 |
| LightGBM | 0.6043 |
| Extra Trees | 0.5935 |
| HistGradientBoosting | 0.5650 |
| Random Forest | 0.5624 |

CatBoost produced the strongest holdout R² among the tested models.

## 🏆 Final Model

The final production model is:

**CatBoostRegressor**

### Configuration:
- **Iterations**: 800
- **Learning Rate**: 0.03
- **Depth**: 6
- **L2 Leaf Regularization**: 5
- **Random Seed**: 42

The trained model is stored as:
```
models/bigmart_catboost_final.cbm
```

## 📈 Model Performance

The final model achieved the following holdout results:

| Metric | Result |
|--------|--------|
| R² | 0.6174 |
| MAE | 713.94 |
| RMSE | 1019.76 |
| Within ±10% | 16.13% |
| Within ±20% | 34.90% |

The 5-fold out-of-fold evaluation produced an R² of approximately:
```
0.5995
```

### Important
R² is not the same thing as classification accuracy.

An R² of 0.6174 means that approximately 61.74% of the variance in the target was explained by the model on the holdout evaluation.

## 🔬 Model Selection

CatBoost was selected because it produced the strongest validation performance among the tested models while naturally handling categorical variables.

Additional experiments were performed with:

- Feature engineering variations
- Missing-value strategies
- Log-transformed target
- XGBoost
- LightGBM
- Extra Trees
- HistGradientBoosting
- Random Forest
- Ensemble weighting
- Stacking
- Hyperparameter optimization

Some experiments did not improve the validation performance and were therefore not selected for production.

This is an important part of the modeling process: not every experiment improves the final model.

## 🧠 Explainable AI with SHAP

SHAP was integrated into the application to explain individual predictions.

For each prediction, the application displays:

- SHAP waterfall visualization
- Feature contribution values
- Features increasing the prediction
- Features decreasing the prediction

### Interpretation Guide:

| SHAP Value | Effect |
|------------|--------|
| Positive SHAP value | Pushes prediction upward |
| Negative SHAP value | Pushes prediction downward |

This provides local interpretability for individual predictions.

## 🖥️ Streamlit Application

The application provides three major sections.

### 🔮 Single Prediction

Users can enter product and outlet information and receive:

- Predicted sales
- Model information
- SHAP explanation
- Feature contribution analysis

### 📂 Batch Prediction

Users can upload a CSV file and generate predictions for multiple observations.

The application provides:

- Input validation
- Prediction results
- Number of rows
- Total predicted sales
- Average predicted sales
- Downloadable prediction CSV

### 📊 Model Information

The application exposes:

- Model name
- R²
- MAE
- RMSE
- Model configuration
- Number of features

## ⚡ FastAPI Backend

The machine learning model is exposed through a REST API.

### Available Endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/` | Root endpoint |
| GET | `/model-info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict-batch` | Batch prediction |

### Health Check
**GET** `/health`

Used to verify that the API service is available.

### Model Information
**GET** `/model-info`

Returns model configuration and evaluation information.

### Single Prediction
**POST** `/predict`

Accepts one observation and returns the predicted sales.

### Batch Prediction
**POST** `/predict-batch`

Accepts multiple observations and returns predictions for each row.

FastAPI also uses Pydantic validation to reject invalid inputs before they reach the model.


## 🏗️ System Architecture

```
                    USER
                      │
                      ▼
            ┌───────────────────┐
            │   Streamlit UI    │
            │                   │
            │ Single Prediction │
            │ Batch Prediction  │
            │ SHAP Explanation  │
            └─────────┬─────────┘
                      │
                  HTTPS / JSON
                      │
                      ▼
            ┌───────────────────┐
            │      FastAPI      │
            │                   │
            │ /health           │
            │ /model-info       │
            │ /predict          │
            │ /predict-batch    │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ Prediction        │
            │ Pipeline          │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ CatBoost Model    │
            │ .cbm              │
            └───────────────────┘
```

The application follows a three-tier architecture:
- **Frontend**: Streamlit UI for user interaction
- **Backend**: FastAPI REST API for predictions
- **Model**: CatBoost regression model with SHAP explainability

## 📁 Project Structure

```
BigMart-Sales-ML/
│
├── api/
│   └── main.py
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│   ├── bigmart_catboost_final.cbm
│   ├── bigmart_feature_metadata.json
│   └── bigmart_model_metadata.json
│
├── notebooks/
│   └── BigMart_Sales_Advanced.ipynb
│
├── reports/
│
├── src/
│   └── prediction_pipeline.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

## 🔎 Exploratory Data Analysis

EDA was performed to understand:

- Dataset structure
- Missing values
- Duplicate records
- Unique values
- Numerical distributions
- Categorical distributions
- Outliers
- Target distribution
- Feature correlations
- Feature-target relationships

The goal of EDA was to identify data-quality issues and discover useful patterns before model development.

## 🛠️ Feature Engineering

Several additional features were created to improve the representation of the underlying business problem.

Examples include:

### Outlet Age
```
Outlet_Age = Current Year - Outlet_Establishment_Year
```

### Item Identifier Category
The first two characters of the product identifier were extracted as a higher-level product category.

### Zero Visibility Indicator
`Item_Visibility_Zero` identifies products with zero recorded visibility.

### Log-transformed Visibility
`Item_Visibility_Log` was created using:
```python
np.log1p(Item_Visibility)
```

### Log-transformed MRP
`Item_MRP_Log` was created using:
```python
np.log1p(Item_MRP)
```

## 🤖 Models Evaluated

Multiple regression algorithms were evaluated under the same validation framework.

| Model | Validation R² |
|-------|---------------|
| CatBoost | 0.6174 |
| XGBoost | 0.6115 |
| LightGBM | 0.6043 |
| Extra Trees | 0.5935 |
| HistGradientBoosting | 0.5650 |
| Random Forest | 0.5624 |

CatBoost produced the strongest holdout R² among the tested models.

## 🏆 Final Model

The final production model is:

**CatBoostRegressor**

### Configuration:
- **Iterations**: 800
- **Learning Rate**: 0.03
- **Depth**: 6
- **L2 Leaf Regularization**: 5
- **Random Seed**: 42

The trained model is stored as:
```
models/bigmart_catboost_final.cbm
```

## 📈 Model Performance

The final model achieved the following holdout results:

| Metric | Result |
|--------|--------|
| R² | 0.6174 |
| MAE | 713.94 |
| RMSE | 1019.76 |
| Within ±10% | 16.13% |
| Within ±20% | 34.90% |

The 5-fold out-of-fold evaluation produced an R² of approximately:
```
0.5995
```

### Important
R² is not the same thing as classification accuracy.

An R² of 0.6174 means that approximately 61.74% of the variance in the target was explained by the model on the holdout evaluation.

## 🔬 Model Selection

CatBoost was selected because it produced the strongest validation performance among the tested models while naturally handling categorical variables.

Additional experiments were performed with:

- Feature engineering variations
- Missing-value strategies
- Log-transformed target
- XGBoost
- LightGBM
- Extra Trees
- HistGradientBoosting
- Random Forest
- Ensemble weighting
- Stacking
- Hyperparameter optimization

Some experiments did not improve the validation performance and were therefore not selected for production.

This is an important part of the modeling process: not every experiment improves the final model.

## 🧠 Explainable AI with SHAP

SHAP was integrated into the application to explain individual predictions.

For each prediction, the application displays:

- SHAP waterfall visualization
- Feature contribution values
- Features increasing the prediction
- Features decreasing the prediction

### Interpretation Guide:

| SHAP Value | Effect |
|------------|--------|
| Positive SHAP value | Pushes prediction upward |
| Negative SHAP value | Pushes prediction downward |

This provides local interpretability for individual predictions.

## 🖥️ Streamlit Application

The application provides three major sections.

### 🔮 Single Prediction

Users can enter product and outlet information and receive:

- Predicted sales
- Model information
- SHAP explanation
- Feature contribution analysis

### 📂 Batch Prediction

Users can upload a CSV file and generate predictions for multiple observations.

The application provides:

- Input validation
- Prediction results
- Number of rows
- Total predicted sales
- Average predicted sales
- Downloadable prediction CSV

### 📊 Model Information

The application exposes:

- Model name
- R²
- MAE
- RMSE
- Model configuration
- Number of features

## ⚡ FastAPI Backend

The machine learning model is exposed through a REST API.

### Available Endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/` | Root endpoint |
| GET | `/model-info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict-batch` | Batch prediction |

### Health Check
**GET** `/health`

Used to verify that the API service is available.

### Model Information
**GET** `/model-info`

Returns model configuration and evaluation information.

### Single Prediction
**POST** `/predict`

Accepts one observation and returns the predicted sales.

### Batch Prediction
**POST** `/predict-batch`

Accepts multiple observations and returns predictions for each row.

FastAPI also uses Pydantic validation to reject invalid inputs before they reach the model.

## 🐳 Docker

The application is containerized using Docker.

Separate Docker images are used for:

- **FastAPI** - Backend REST API service
- **Streamlit** - Frontend web application

### Docker Files:

- `Dockerfile.api` - FastAPI container configuration
- `Dockerfile.streamlit` - Streamlit container configuration
- `docker-compose.yml` - Multi-container orchestration

### Build and Run:

```bash
docker compose build
docker compose up
```

The services will be available at:
- **Streamlit UI**: http://localhost:8501
- **FastAPI**: http://localhost:8000

## 🚀 Local Setup

### Clone the repository:

```bash
git clone https://github.com/Ankitkumar-x/BigMart-Sales-ML.git
cd BigMart-Sales-ML
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run FastAPI Locally

```bash
python -m uvicorn api.main:app --reload
```

Access the API at:
- **API Root**: http://127.0.0.1:8000
- **Swagger Docs**: http://127.0.0.1:8000/docs

## ▶️ Run Streamlit Locally

```bash
streamlit run app/app.py
```

Access the application at:
- **Application**: http://localhost:8501

## ☁️ Deployment

The project is deployed using Render.

### Frontend - Streamlit

**Live URL**: https://bigmart-sales-ml.onrender.com

Streamlit frontend with full UI for predictions and model information.

### Backend - FastAPI

**Live URL**: https://sales-v8m6.onrender.com

FastAPI backend serving predictions and model metadata.

The Streamlit frontend communicates with the FastAPI backend through the `API_URL` environment variable.

## 🔐 Input Validation

The application validates incoming values before prediction.

### Validation Examples:

- Positive item weight
- Non-negative item visibility
- Positive item MRP
- Valid outlet establishment year
- Required CSV columns
- Empty CSV detection

FastAPI additionally validates request payloads using Pydantic schemas.

## ⚠️ Limitations

The current model is based on the information available in the BigMart dataset.

Important real-world variables such as:

- Promotions
- Discounts
- Historical sales
- Seasonal effects
- Store traffic
- Inventory levels
- Competitor information

are not available in the dataset.

Therefore, model performance is limited by the information contained in the available features.

The current holdout R² is approximately:

```
0.6174
```

and should not be presented as "61.74% accuracy."

## 🔮 Future Improvements

Potential future improvements include:

- Additional external business features
- Time-series and seasonal information
- Historical outlet-level sales
- Promotion and discount features
- More advanced feature selection
- Neural-network experiments
- Model monitoring
- Prediction logging
- API authentication
- Rate limiting
- Automated CI/CD
- Cloud-native monitoring

## 💼 Summary

This project demonstrates an end-to-end machine learning workflow:

```
Data
 ↓
EDA
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Model Comparison
 ↓
Cross Validation
 ↓
Hyperparameter Optimization
 ↓
Ensemble / Stacking Experiments
 ↓
CatBoost Selection
 ↓
SHAP Explainability
 ↓
Streamlit UI
 ↓
FastAPI Backend
 ↓
Docker Containerization
 ↓
GitHub Repository
 ↓
Cloud Deployment
```

The main focus was not only on model training but also on building a reproducible and deployable machine learning system.

## 👨‍💻 Author

**Ankit Kumar**

GitHub: https://github.com/Ankitkumar-x
