"""Dataset loading and cleaning utilities for AutoPulse."""
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "name", "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CarDekho CSV and validate its expected columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the row index."""
    cleaned = df.drop_duplicates().copy()
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def save_cleaned_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the cleaned dataframe, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def preprocess_raw_dataset(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load raw data, remove duplicates, save and return the cleaned data."""
    raw_df = load_dataset(raw_path)
    cleaned_df = clean_dataset(raw_df)
    save_cleaned_dataset(cleaned_df, output_path)
    return cleaned_df
