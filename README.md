
# Rossmann Sales Prediction and Store Performance Analysis

## Introduction

Rossmann, a leading European drugstore chain founded in 1972, operates over 4000 stores in countries like Germany, Poland, and Hungary. This project is based on a Kaggle competition focused on predicting Rossmann store sales using time series data. The goal is to perform forecasting to anticipate future sales and optimize commercial planning.

This comprehensive analysis leverages advanced machine learning techniques and focuses on key objectives:
- **Preprocessing & Feature Engineering**: Extracting additional insights and generating new features.
- **Machine Learning Models**: Comparing Random Forest and XGBoost for forecasting.
- **Hyperparameter Optimization**: Employing robust techniques like Time Series Cross-Validation.
- **Predictions & Validation**: Evaluating models using RMSPE and interpreting results.

Additionally, a secondary problem classifies store performance into "High", "Medium", and "Low" categories based on sales percentiles, providing actionable insights.

---

## Exploratory Data Analysis

### Key Dataset Features
The dataset includes variables such as store type, assortment, promotions, and competition distance, among others.

#### Notable Insights:
- **Sales by Store Type and Assortment**: Store Type 'b' consistently outperforms others, likely due to effective product offerings or marketing strategies.
- **Customers and Sales Correlation**: A strong positive relationship indicates strategies to increase customer traffic could boost sales.
- **Competition Distance and Sales**: Stores closer to competitors often perform better, potentially due to higher population density and competitive marketing.

---

## Feature Engineering
Key transformations include:
1. **Date Features**: Extracting day, month, year, and other time-based variables.
2. **Categorical Encoding**: Mapping and dummy variable creation for `Assortment`, `StateHoliday`, etc.
3. **Promo Features**: Extracting active promotion months into binary flags.
4. **Handling Missing Values**: Replacing missing data in competition and promotion-related features with -1.
5. **Feature Reduction**: Removing redundant columns after dummy creation.

---

## Machine Learning Models

### Random Forest and XGBoost
- **Benchmark Model**: XGBoost was used as a baseline.
- **Primary Model**: Random Forest with hyperparameter optimization.
- **Metric**: RMSPE (Root Mean Squared Percentage Error).

#### Optimization:
- Hyperparameter tuning through Random Search and Time Series Cross-Validation.
- Final RMSPE on test data: **0.11833** (Random Forest) vs. **0.12970** (XGBoost).

---

## Secondary Machine Learning Task: Store Performance Classification

Stores are classified into "High", "Medium", and "Low" performance categories using a CatBoost classifier, evaluated with ROC-AUC (score: **0.9453**). SHAP values provide insights into feature importance:
- **Key Features**: Competition distance, promotions, store type, and specific calendar dates.

---

## Visual Insights
Key visualizations include:
- Sales trends across store types and assortments.
- Customer-sales scatter plots.
- SHAP value explanations for feature contributions.

---

## Results
This project successfully demonstrates how machine learning models can predict sales and classify store performance, offering actionable insights for business strategy. The methods and results are validated through competitive Kaggle scores and robust interpretability techniques.
