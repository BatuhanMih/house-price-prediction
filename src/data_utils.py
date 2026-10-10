"""Shared data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "emlak_fiyat.xlsx"

TARGET_COLUMN = "Fiyat (TL)"
CATEGORICAL_FEATURES = ["Konum", "Oda Sayısı", "Asansör", "Doğal Gaz"]
NUMERICAL_FEATURES = ["Metrekare", "Bina Yaşı", "Kat Durumu"]
# The original workbook contains replacement characters in a few Turkish labels.
# These aliases let the public training scripts work with both the archived file
# and a corrected copy of the dataset.
COLUMN_ALIASES = {
    "Oda Say�s�": "Oda Sayısı",
    "Bina Ya��": "Bina Yaşı",
    "Asans�r": "Asansör",
    "Do�al Gaz": "Doğal Gaz",
}

VALUE_ALIASES = {
    "G�rele": "Görele",
    "Ke�ap": "Keşap",
    "�ebinkarahisar": "Şebinkarahisar",
    "Hay�r": "Hayır",
}


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Excel dataset, repair known labels, and validate its schema."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_excel(data_path).rename(columns=COLUMN_ALIASES)
    data = data.replace(VALUE_ALIASES)

    required = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data[required].copy()


def split_features_target(data: pd.DataFrame):
    """Return model inputs and the house-price target."""
    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN]


def build_preprocessor(*, scale_numeric: bool) -> "ColumnTransformer":
    """Create a reusable transformer for numeric and categorical features."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERICAL_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
