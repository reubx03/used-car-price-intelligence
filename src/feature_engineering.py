"""Feature engineering and preprocessing pipeline construction."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = ["year", "km_driven"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner", "brand"]
TARGET = "selling_price"


def add_brand_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the first token of the vehicle name as its brand."""
    result = df.copy()
    if "name" not in result.columns:
        if "brand" in result.columns:
            return result
        raise ValueError("Input data must contain a 'name' column.")
    result["brand"] = result["name"].astype(str).str.split().str[0]
    result.drop(columns=["name"], inplace=True)
    return result


def build_preprocessor() -> ColumnTransformer:
    """Create the same numeric/categorical transformer as the notebook."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
