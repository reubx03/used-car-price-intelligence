from pathlib import Path
import pandas as pd
from src.data_preprocessing import clean_dataset
from src.feature_engineering import add_brand_feature, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_cleaned_dataset_shape_and_quality():
    df = pd.read_csv(ROOT / "data/processed/used_cars_cleaned.csv")
    assert df.shape == (3577, 8)
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_clean_dataset_removes_exact_duplicates():
    sample = pd.DataFrame({"name": ["A", "A", "B"], "year": [2010, 2010, 2012]})
    cleaned = clean_dataset(sample)
    assert len(cleaned) == 2
    assert cleaned.duplicated().sum() == 0


def test_brand_feature_matches_training_schema():
    df = pd.read_csv(ROOT / "data/processed/used_cars_cleaned.csv").head(5)
    transformed = add_brand_feature(df)
    assert "name" not in transformed.columns
    assert "brand" in transformed.columns
    assert transformed.iloc[0]["brand"] == "Maruti"


def test_preprocessor_can_be_created():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
