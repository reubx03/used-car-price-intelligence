# AutoPulse — Used Car Price Intelligence

**An interactive machine learning application for estimating used-car selling prices.**

AutoPulse uses supervised machine learning to predict used-car prices based on vehicle specifications, usage, and ownership details. It combines exploratory data analysis, regression modelling, and interactive data visualization in a Streamlit application.

## Overview

The objective of this project is to develop a data-driven system that estimates the selling price of a used vehicle using its key characteristics.

The project follows a complete machine learning workflow, from data exploration and preprocessing to model development, evaluation, and application development.

## Key Features

- **Price Prediction:** Estimate used-car selling prices using a trained Random Forest model.
- **Market Insights:** Explore vehicle price distributions and market patterns.
- **Interactive Visualizations:** Analyze trends using Plotly charts.
- **Model Comparison:** Evaluate and compare multiple regression algorithms.
- **Interactive Application:** Access predictions and insights through Streamlit.

## Dataset

| Attribute | Details |
|---|---|
| Original Records | 4,340 |
| Cleaned Records | 3,577 |
| Features | 8 original columns |
| Missing Values | 0 |
| Duplicate Records Remaining | 0 |
| Target Variable | `selling_price` |

The dataset contains information about vehicle names, manufacturing years, selling prices, kilometres driven, fuel types, seller types, transmission types, and ownership history.

Feature engineering was performed to extract vehicle brands from the original vehicle names.

## Machine Learning

Three regression algorithms were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Model Performance

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Linear Regression | 179,642.89 | 384,722.69 | 0.54 |
| Decision Tree | 198,734.06 | 473,873.48 | 0.30 |
| **Random Forest** | **158,827.69** | **367,979.71** | **0.58** |

**Selected Model: Random Forest Regressor**

Random Forest achieved the best overall results among the three evaluated models, with the lowest MAE and RMSE and the highest R² score.

*These are the initial evaluation results. Further cross-validation and hyperparameter tuning are planned.*

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Application | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/reubx03/used-car-price-intelligence.git
cd used-car-price-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal to access AutoPulse.

## Project Structure

```text
used-car-price-intelligence/
│
├── app/                 # Streamlit application
├── data/
│   ├── raw/             # Original dataset
│   └── processed/       # Cleaned dataset
├── models/              # Trained ML models
├── notebooks/           # EDA and ML experiments
├── reports/             # Project reports
├── src/                 # Supporting source code
├── .streamlit/          # Application configuration
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Future Improvements

- Cross-validation and hyperparameter optimization.
- Further model performance improvements.
- Application refinement and deployment.

## Author

**Reuben Mathew Tharakan**

B.Tech — Artificial Intelligence & Data Science

[GitHub Profile](https://github.com/reubx03) · [Project Repository](https://github.com/reubx03/used-car-price-intelligence)

---

*Developed as a machine learning regression project using Python and Streamlit.*
