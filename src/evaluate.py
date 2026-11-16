"""Evaluation helpers for regression models."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(model_name, y_true, y_pred) -> dict:
    """Calculate MAE, RMSE and R-squared for one model."""
    return {
        "Model": model_name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2 Score": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(predictions: dict, y_true) -> pd.DataFrame:
    """Evaluate {model_name: predictions} and return a comparison table."""
    rows = [
        evaluate_regression_model(name, y_true, pred)
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
