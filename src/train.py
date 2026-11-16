"""Train and evaluate AutoPulse regression models.

Run from the repository root:
    python -m src.train

To regenerate the cleaned CSV from the original dataset first:
    python -m src.train --from-raw
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import preprocess_raw_dataset
from src.feature_engineering import add_brand_feature, build_preprocessor, TARGET
from src.evaluate import evaluate_models

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "CAR DETAILS FROM CAR DEKHO.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "used_cars_cleaned.csv"
MODEL_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "metrics" / "model_comparison.csv"


def train_and_evaluate(df: pd.DataFrame):
    """Train three models using the notebook's 80/20 split and seed 42."""
    model_df = add_brand_feature(df)
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_definitions = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    fitted_models = {}
    predictions = {}
    for name, estimator in model_definitions.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        predictions[name] = pipeline.predict(X_test)

    metrics = evaluate_models(predictions, y_test)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    return fitted_models, metrics, best_name, best_model


def main():
    parser = argparse.ArgumentParser(description="Train AutoPulse ML models")
    parser.add_argument(
        "--from-raw", action="store_true",
        help="Recreate the cleaned CSV from data/raw before training."
    )
    args = parser.parse_args()

    if args.from_raw:
        if not RAW_PATH.exists():
            raise FileNotFoundError(
                f"Original dataset missing: {RAW_PATH}\n"
                "Download 'CAR DETAILS FROM CAR DEKHO.csv' and place it in data/raw/."
            )
        df = preprocess_raw_dataset(RAW_PATH, CLEAN_PATH)
    else:
        if not CLEAN_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset missing: {CLEAN_PATH}")
        df = pd.read_csv(CLEAN_PATH)

    fitted, metrics, best_name, best_model = train_and_evaluate(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
    }
    for name, filename in filenames.items():
        joblib.dump(fitted[name], MODEL_DIR / filename)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(f"Dataset shape: {df.shape}")
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nSelected model: {best_name}")
    print(f"Saved models to: {MODEL_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
