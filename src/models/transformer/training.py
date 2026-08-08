import json

from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
import tensorflow as tf

# Training runs locally on CPU (no Colab GPU), so cap intra-op parallelism
# at import time -- before any model gets built -- so a run leaves the rest
# of the machine usable instead of claiming every core.
tf.config.threading.set_intra_op_parallelism_threads(8)

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.common import data
from src.models.common.evaluation import evaluate_and_save
from src.models.common.plotting import plot_forecast_example, plot_loss_curve, plot_mae_by_horizon, plot_mae_curve
from src.models.transformer import architecture


@dataclass
class TransformerConfig:
    input_hours: int = data.INPUT_HOURS
    output_hours: int = data.OUTPUT_HOURS
    stride: int = data.STRIDE
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    ff_dim: int = 128
    dropout_rate: float = 0.20
    batch_size: int = 64
    epochs: int = 25
    baseline_blend: bool = False

    def to_dict(self):
        return asdict(self)


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_training(config, experiment_dir, checkpoint_path, model=None, show_plots=False,
                  feature_columns=data.FEATURE_COLUMNS, target_column=data.TARGET_COLUMN,
                  load_splits_fn=data.load_splits,
                  city_aware=False, city_vocab=data.CITY_VOCAB,
                  city_embed_dim=8, experiment_name=None):
    """Loads data, builds (or reuses a supplied) model, trains it with early
    stopping, evaluates on the test set, and saves every experiment artifact.
    Returns a results dict shaped the same as load_cached_results(), so
    callers don't need to branch on whether training actually happened.

    `feature_columns`/`target_column`/`load_splits_fn` let experiments swap
    in different inputs (e.g. extra features, a city-filtered dataset)
    without duplicating this whole training/eval/save loop.

    Windows are always built with city ids (data.create_multistep_sequences(
    ..., with_city_ids=True)), for every experiment regardless of
    `city_aware` -- the per-city breakdown in evaluate_and_save is wanted
    whether or not the model itself consumes city identity as a feature.

    `city_aware`, when True, builds a two-input model (weather sequence +
    city id) and fits/evaluates on [X, city_ids] instead of a bare array.
    Left at its default of False, the model only ever sees the weather
    sequence; city ids are still built and still used for the per-city
    breakdown.

    `config.baseline_blend` is threaded through to architecture.build_model(),
    which adds the last observed raw temperature to every forecast hour.
    It's rejected with a ValueError there unless target_column is actually
    one of feature_columns (see build_model's docstring). It comes from the
    config -- not a separate run_training() parameter -- so config.json
    always describes the model that was actually built; nothing here can
    disagree with it.

    `experiment_name` labels the model series in the MAE-by-horizon plot;
    if not given, it's inferred from experiment_dir's grandparent directory
    name (.../Transformer/<experiment_name>/experiments/<timestamp>/)."""

    if experiment_name is None:
        experiment_name = experiment_dir.parent.parent.name

    figure_dir = experiment_dir / "figures"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_splits_fn()

    X_train, y_train, city_train = data.create_multistep_sequences(
        train_df, config.input_hours, config.output_hours, config.stride,
        feature_columns, target_column, with_city_ids=True
    )
    X_dev, y_dev, city_dev = data.create_multistep_sequences(
        dev_df, config.input_hours, config.output_hours, config.stride,
        feature_columns, target_column, with_city_ids=True
    )
    X_test, y_test, city_test = data.create_multistep_sequences(
        test_df, config.input_hours, config.output_hours, config.stride,
        feature_columns, target_column, with_city_ids=True
    )

    # Rebuilt from a single-column frame so persistence/previous-day
    # baselines score the target's own raw history, regardless of whether
    # feature_columns even includes it (exp00/exp02 don't). This stays
    # aligned to X_test/y_test because segmentation and stride depend only
    # on the frame and the window sizes, not the feature set.
    temperature_history = data.create_multistep_sequences(
        test_df, config.input_hours, config.output_hours, config.stride,
        feature_columns=[data.TARGET_COLUMN]
    )[0][:, :, 0]

    if model is None:
        normalizer = layers.Normalization()
        normalizer.adapt(X_train.reshape(-1, X_train.shape[-1]))

        model = architecture.build_model(
            window_size=config.input_hours,
            num_features=X_train.shape[-1],
            output_hours=config.output_hours,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
            normalizer=normalizer,
            num_cities=len(city_vocab) if city_aware else None,
            city_embed_dim=city_embed_dim,
            baseline_blend=config.baseline_blend,
            temperature_in_features=(target_column in feature_columns)
        )

    model.summary()

    with open(experiment_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

    metrics, predictions = evaluate_and_save(
        model, X_test, y_test, temperature_history,
        experiment_dir / "metrics.json",
        experiment_dir / "predictions_sample.csv",
        model_input=fit_X_test,
        city_ids=city_test,
        feature_columns=feature_columns,
        stride=config.stride,
    )

    plot_loss_curve(history, figure_dir, show=show_plots)
    plot_mae_curve(history, figure_dir, show=show_plots)
    plot_forecast_example(y_test, predictions, figure_dir, show=show_plots)
    plot_mae_by_horizon(
        metrics["improved_model"],
        {
            "Persistence": metrics["persistence_baseline"],
            "Previous day": metrics["previous_day_baseline"],
        },
        figure_dir, model_label=experiment_name, show=show_plots
    )

    model.save(experiment_dir / "weather_transformer.keras")

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    return {
        "config": config.to_dict(),
        "experiment_dir": experiment_dir,
        "metrics": metrics,
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
    hit and a fresh training run return the same shape to the caller.

    y_test/predictions are reconstructed from predictions_sample.csv (a
    few hundred windows, not the full test set -- see evaluate_and_save),
    since the full (n, output_hours) arrays are never persisted. That's
    still enough to re-render plot_forecast_example; plot_mae_by_horizon
    only needs the metrics dict, not the raw arrays."""

    with open(experiment_dir / "metrics.json") as f:
        metrics = json.load(f)

    with open(experiment_dir / "config.json") as f:
        config = json.load(f)

    history_df = pd.read_csv(experiment_dir / "training_history.csv")
    model = tf.keras.models.load_model(experiment_dir / "weather_transformer.keras")

    sample_df = pd.read_csv(experiment_dir / "predictions_sample.csv").sort_values(["window", "hour"])
    y_test = sample_df.pivot(index="window", columns="hour", values="actual").to_numpy()
    predictions = sample_df.pivot(index="window", columns="hour", values="predicted").to_numpy()

    return {
        "config": config,
        "experiment_dir": experiment_dir,
        "metrics": metrics,
        "epochs_trained": len(history_df),
        "model": model,
        "history": _HistoryLike(history_df.to_dict(orient="list")),
        "y_test": y_test,
        "predictions": predictions
    }
