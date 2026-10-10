"""Train and evaluate a configurable multilayer perceptron model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from src.data_utils import DEFAULT_DATA_PATH, build_preprocessor, load_dataset, split_features_target


def parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layers must be comma-separated integers.") from exc
    if not layers or any(size <= 0 for size in layers):
        raise argparse.ArgumentTypeError("At least one positive layer size is required.")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mlp"))
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("64,32"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    data = load_dataset(args.data)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    preprocessor = build_preprocessor(scale_numeric=True)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

    model = Sequential([Input(shape=(x_train_processed.shape[1],))])
    for units in args.layers:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    history = model.fit(
        x_train_processed,
        y_train_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=1,
    )

    predictions_scaled = model.predict(x_test_processed, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_scaled).ravel()
    metrics = {
        "model": "MLP",
        "hidden_layers": args.layers,
        "epochs_completed": len(history.history["loss"]),
        "rows": len(data),
        "test_rows": len(y_test),
        "random_state": args.random_state,
        "mae_tl": float(mean_absolute_error(y_test, predictions)),
        "rmse_tl": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model.keras")
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")
    joblib.dump(target_scaler, args.output_dir / "target_scaler.joblib")
    pd.DataFrame(history.history).to_csv(args.output_dir / "training_history.csv", index=False)
    pd.DataFrame({"actual_tl": y_test, "predicted_tl": predictions}).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
