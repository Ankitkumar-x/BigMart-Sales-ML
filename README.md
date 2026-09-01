# BigMart Sales ML

A machine learning project for sales prediction using advanced regression models and explainable AI.

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
