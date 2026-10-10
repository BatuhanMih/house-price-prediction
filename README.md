# House Price Prediction

This project compares an XGBoost regressor and a multilayer perceptron (MLP) for estimating residential sale prices from tabular property features. It was developed as a three-person student project.

The repository contains training scripts and data-loading tests. The original Excel dataset is not included in this public version because its collection source and redistribution rights have not been documented. The scripts can be run with a dataset you are permitted to use.

## Models and evaluation

- XGBoost: an 80/20 train/test split, with five-fold cross-validation on the training portion before fitting the final model.
- MLP: the same default train/test split, with a validation subset drawn from the training portion for early stopping. Numeric inputs and the target are scaled using training data only.
- Both scripts report MAE, RMSE and R² on their held-out test portion and save predictions locally under `artifacts/`.

The default split seed is 24. The two scripts use different model-selection procedures, so their outputs should not be interpreted as a controlled comparison of model families. Earlier project screenshots are not a reproducible benchmark for these scripts and are not included here.

## Input data

Provide an `.xlsx` file with these columns:

| Column | Meaning |
|---|---|
| `Konum` | District or location |
| `Metrekare` | Floor area in square metres |
| `Oda Sayısı` | Room configuration |
| `Bina Yaşı` | Building age |
| `Kat Durumu` | Floor number or status |
| `Asansör` | Elevator availability |
| `Doğal Gaz` | Natural-gas availability |
| `Fiyat (TL)` | Sale price in Turkish lira |

The loader also recognizes a few damaged Turkish labels found in the archived workbook. It checks the required columns but does not establish the reliability or legal status of a new dataset. Do not upload personal or third-party listing data without permission.

## Run locally

Use Python 3.10 or 3.11 and create a virtual environment. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Train XGBoost:

```bash
python -m src.train_xgboost --data path/to/your_data.xlsx
```

Train the MLP:

```bash
python -m src.train_mlp --data path/to/your_data.xlsx
```

Use `--help` on either command for the available settings. The default input path, if `--data` is omitted, is `data/emlak_fiyat.xlsx`. That file is deliberately ignored by Git.

To run the lightweight loader tests:

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
```

Training creates model and prediction files in `artifacts/`; those files are also ignored by Git.

## Limitations

The archived project used 500 rows and included 11 exact duplicate rows. Both a small sample and duplicate records can make holdout estimates unstable or optimistic. No model here should be used as a production appraisal system without a documented dataset, stronger validation and domain review.

## Team:

- [Batuhan Mıh](https://github.com/BatuhanMih)
- [Mustafa Kurtulan](https://github.com/MustafaKurtulan)
- [Faruk Topal](https://github.com/faruktopal)

## Reuse

No software license has been granted for this repository. Contact the project contributors before reusing or redistributing the code. A license can be added after all contributors agree to its terms.
