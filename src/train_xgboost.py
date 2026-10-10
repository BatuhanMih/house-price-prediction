"""Train and evaluate the XGBoost house-price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_utils import DEFAULT_DATA_PATH, build_preprocessor, load_dataset, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgboost"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=24)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--cv", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=args.estimators,
                    random_state=args.random_state,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                ),
            ),
        ]
    )

    cv_rmse = np.sqrt(
        -cross_val_score(
            model,
            x_train,
            y_train,
            cv=args.cv,
            scoring="neg_mean_squared_error",
        )
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "model": "XGBoost",
        "rows": len(data),
        "test_rows": len(y_test),
        "random_state": args.random_state,
        "mae_tl": float(mean_absolute_error(y_test, predictions)),
        "rmse_tl": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "cv_rmse_tl_mean": float(cv_rmse.mean()),
        "cv_rmse_tl_std": float(cv_rmse.std()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "model.joblib")
    pd.DataFrame({"actual_tl": y_test, "predicted_tl": predictions}).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
