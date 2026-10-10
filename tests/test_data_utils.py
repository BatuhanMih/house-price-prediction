import pandas as pd
import pytest

from src.data_utils import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)


def test_loader_repairs_archived_labels(tmp_path):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame(
        {
            "Konum": ["G�rele"],
            "Metrekare": [100],
            "Oda Say�s�": ["3+1"],
            "Bina Ya��": [5],
            "Kat Durumu": [2],
            "Asans�r": ["Evet"],
            "Do�al Gaz": ["Evet"],
            "Fiyat (TL)": [2_000_000],
        }
    ).to_excel(path, index=False)

    data = load_dataset(path)

    assert list(data.columns) == CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN]
    assert data.loc[0, "Konum"] == "Görele"
    assert data.loc[0, TARGET_COLUMN] == 2_000_000


def test_loader_rejects_missing_required_column(tmp_path):
    path = tmp_path / "missing.xlsx"
    pd.DataFrame({"Konum": ["Merkez"]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
