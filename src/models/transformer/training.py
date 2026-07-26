import json

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.common import data
from src.models.common.evaluation import evaluate_and_save
from src.models.common.plotting import plot_loss_curve, plot_mae_curve, plot_prediction_curve
from src.models.transformer import architecture


@dataclass
class TransformerConfig:
    window_size: int = data.WINDOW_SIZE
    forecast_horizon: int = data.FORECAST_HORIZON
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    ff_dim: int = 128
    dropout_rate: float = 0.20
    batch_size: int = 32
    epochs: int = 25

    def to_dict(self):
        return asdict(self)


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_training(config, experiment_dir, checkpoint_path, model=None, show_plots=False,
                  feature_columns=data.FEATURE_COLUMNS, target_column=data.TARGET_COLUMN,
                  load_splits_fn=data.load_splits,
                  city_aware=False, city_column=data.CITY_COLUMN, city_vocab=data.CITY_VOCAB,
                  city_embed_dim=8):
    """Loads data, builds (or reuses a supplied) model, trains it with early
    stopping, evaluates on the test set, and saves every experiment artifact.
    Returns a results dict shaped the same as load_cached_results(), so
    callers don't need to branch on whether training actually happened.

    `feature_columns`/`target_column`/`load_splits_fn` let experiments swap
    in different inputs (e.g. extra features, a city-filtered dataset)
    without duplicating this whole training/eval/save loop.

    `city_aware`, when True, builds windows with data.create_city_aware_sequences()
    instead of data.create_sequences() (so a window never spans two cities),
    builds a two-input model (weather sequence + city id), and fits/evaluates
    on [X, city_ids] instead of a bare array. Left at its default of False,
    this function's behavior is unchanged."""

    figure_dir = experiment_dir / "figures"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_splits_fn()

    if city_aware:
        X_train, y_train, city_train = data.create_city_aware_sequences(
            train_df, config.window_size, config.forecast_horizon, feature_columns, target_column,
            city_column, city_vocab
        )
        X_dev, y_dev, city_dev = data.create_city_aware_sequences(
            dev_df, config.window_size, config.forecast_horizon, feature_columns, target_column,
            city_column, city_vocab
        )
        X_test, y_test, city_test = data.create_city_aware_sequences(
            test_df, config.window_size, config.forecast_horizon, feature_columns, target_column,
            city_column, city_vocab
        )
    else:
        X_train, y_train = data.create_sequences(
            train_df, config.window_size, config.forecast_horizon, feature_columns, target_column
        )
        X_dev, y_dev = data.create_sequences(
            dev_df, config.window_size, config.forecast_horizon, feature_columns, target_column
        )
        X_test, y_test = data.create_sequences(
            test_df, config.window_size, config.forecast_horizon, feature_columns, target_column
        )

    if model is None:
        normalizer = layers.Normalization()
        normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))

        model = architecture.build_model(
            window_size=config.window_size,
            num_features=X_train.shape[-1],
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            normalizer=normalizer,
            num_cities=len(city_vocab) if city_aware else None,
            city_embed_dim=city_embed_dim
        )

    model.summary()

    with open(experiment_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    fit_X_train = [X_train, city_train] if city_aware else X_train
    fit_X_dev = [X_dev, city_dev] if city_aware else X_dev
    fit_X_test = [X_test, city_test] if city_aware else X_test

    history = model.fit(
        fit_X_train,
        y_train,
        validation_data=(fit_X_dev, y_dev),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(experiment_dir / "training_history.csv", index=False)

    loss, mae, rmse, predictions = evaluate_and_save(
        model, fit_X_test, y_test,
        experiment_dir / "metrics.json",
        experiment_dir / "predictions.csv"
    )

    plot_loss_curve(history, figure_dir, show=show_plots)
    plot_mae_curve(history, figure_dir, show=show_plots)
    plot_prediction_curve(y_test, predictions, figure_dir, show=show_plots)

    model.save(experiment_dir / "weather_transformer.keras")

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    return {
        "config": config.to_dict(),
        "experiment_dir": experiment_dir,
        "loss": loss,
        "mae": mae,
        "rmse": rmse,
        "epochs_trained": len(history_df),
        "model": model,
        "history": history,
        "y_test": y_test,
        "predictions": predictions
    }


class _HistoryLike:
    """Mimics the subset of tf.keras.callbacks.History that plotting.py
    relies on (a `.history` dict), so a cached run can be re-plotted with
    the exact same plot_*_curve(history, ...) calls as a fresh run."""

    def __init__(self, history_dict):
        self.history = history_dict


def load_cached_results(experiment_dir):
    """Reads back everything run_training() would have produced, so a cache
    hit and a fresh training run return the same shape to the caller."""

    with open(experiment_dir / "metrics.json") as f:
        metrics = json.load(f)

    with open(experiment_dir / "config.json") as f:
        config = json.load(f)

    history_df = pd.read_csv(experiment_dir / "training_history.csv")
    predictions_df = pd.read_csv(experiment_dir / "predictions.csv")
    model = tf.keras.models.load_model(experiment_dir / "weather_transformer.keras")

    return {
        "config": config,
        "experiment_dir": experiment_dir,
        "loss": metrics["loss"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "epochs_trained": len(history_df),
        "model": model,
        "history": _HistoryLike(history_df.to_dict(orient="list")),
        "y_test": predictions_df["Actual"].values,
        "predictions": predictions_df["Predicted"].values
    }
