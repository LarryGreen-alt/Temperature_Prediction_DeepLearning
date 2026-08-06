from __future__ import annotations

import os

# Set logging before TensorFlow is imported, including imports from model.py.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .model import (
        FEATURE_COLUMNS,
        INPUT_HOURS,
        OUTPUT_HOURS,
        TARGET_COLUMN,
        TEMPERATURE_FEATURE_INDEX,
        build_weather_model,
        load_weather_model,
    )
except ImportError:
    from model import (
        FEATURE_COLUMNS,
        INPUT_HOURS,
        OUTPUT_HOURS,
        TARGET_COLUMN,
        TEMPERATURE_FEATURE_INDEX,
        build_weather_model,
        load_weather_model,
    )


RANDOM_SEED = 21
TIME_COLUMN_CANDIDATES = ["time", "datetime", "timestamp", "date"]
GROUP_COLUMN_CANDIDATES = ["city", "location", "location_name"]


@dataclass(frozen=True)
class SequenceMetadata:
    group: str
    segment: int
    start_index: int
    forecast_start: str | None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd(), script_dir, *script_dir.parents]

    for candidate in candidates:
        if (candidate / "data" / "splits").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the project root containing data/splits."
    )


def detect_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    return next(
        (name for name in candidates if name in dataframe.columns),
        None,
    )


def validate_dataframe(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"{split_name} is missing required columns: {missing}"
        )

    dataframe = dataframe.copy()

    for column in FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid = dataframe[FEATURE_COLUMNS].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise ValueError(
            f"{split_name} contains missing/non-numeric values:\n"
            f"{invalid.to_string()}"
        )

    dataframe[FEATURE_COLUMNS] = dataframe[FEATURE_COLUMNS].astype(
        np.float32
    )
    return dataframe


def summarize_split_ranges(
    dataframe: pd.DataFrame,
    split_name: str,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Return each location's timestamp range for leakage checks.

    A trustworthy forecast benchmark must keep each location's training,
    development, and test periods in chronological order.
    """
    time_column = detect_column(dataframe, TIME_COLUMN_CANDIDATES)
    group_column = detect_column(dataframe, GROUP_COLUMN_CANDIDATES)

    if time_column is None:
        raise ValueError(
            f"{split_name} has no recognized time column. "
            "A chronological 2015-2026 split cannot be audited safely."
        )

    working = dataframe.copy()
    working[time_column] = pd.to_datetime(
        working[time_column],
        errors="coerce",
        utc=True,
    )
    if working[time_column].isna().any():
        bad_count = int(working[time_column].isna().sum())
        raise ValueError(
            f"{split_name} has {bad_count} invalid timestamps "
            f"in '{time_column}'."
        )

    if group_column is None:
        grouped = [("all", working)]
    else:
        grouped = working.groupby(
            group_column,
            sort=False,
            dropna=False,
        )

    summary: dict[str, tuple[pd.Timestamp, pd.Timestamp, int]] = {}
    for group_name, group_df in grouped:
        summary[str(group_name)] = (
            group_df[time_column].min(),
            group_df[time_column].max(),
            int(len(group_df)),
        )
    return summary


def audit_chronological_splits(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Fail fast when train/dev/test periods overlap for the same location.
    """
    ranges = {
        "train": summarize_split_ranges(train_df, "training data"),
        "dev": summarize_split_ranges(dev_df, "development data"),
        "test": summarize_split_ranges(test_df, "testing data"),
    }

    common_groups = (
        set(ranges["train"])
        & set(ranges["dev"])
        & set(ranges["test"])
    )
    if not common_groups:
        raise ValueError(
            "No location/group appears in all three splits. "
            "This prevents a comparable chronological evaluation."
        )

    violations: list[str] = []
    print("\nChronological split audit")
    print("-" * 80)

    for group in sorted(common_groups):
        train_start, train_end, train_rows = ranges["train"][group]
        dev_start, dev_end, dev_rows = ranges["dev"][group]
        test_start, test_end, test_rows = ranges["test"][group]

        print(
            f"{group}: "
            f"train {train_start.date()}..{train_end.date()} "
            f"({train_rows:,}) | "
            f"dev {dev_start.date()}..{dev_end.date()} "
            f"({dev_rows:,}) | "
            f"test {test_start.date()}..{test_end.date()} "
            f"({test_rows:,})"
        )

        if train_end >= dev_start:
            violations.append(
                f"{group}: training ends {train_end}, "
                f"but development starts {dev_start}"
            )
        if dev_end >= test_start:
            violations.append(
                f"{group}: development ends {dev_end}, "
                f"but testing starts {test_start}"
            )

    if violations:
        raise ValueError(
            "Chronological split leakage/overlap detected:\n- "
            + "\n- ".join(violations)
        )

    return {
        split_name: {
            group: {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "rows": rows,
            }
            for group, (start, end, rows) in split_ranges.items()
        }
        for split_name, split_ranges in ranges.items()
    }


def split_into_contiguous_groups(
    dataframe: pd.DataFrame,
    split_name: str,
):
    time_column = detect_column(
        dataframe,
        TIME_COLUMN_CANDIDATES,
    )
    group_column = detect_column(
        dataframe,
        GROUP_COLUMN_CANDIDATES,
    )

    working = dataframe.copy()

    if time_column is not None:
        working[time_column] = pd.to_datetime(
            working[time_column],
            errors="coerce",
        )
        bad_count = int(working[time_column].isna().sum())
        if bad_count:
            raise ValueError(
                f"{split_name} has {bad_count} invalid timestamps "
                f"in '{time_column}'."
            )

    if group_column is None:
        grouped = [("all", working)]
    else:
        grouped = working.groupby(
            group_column,
            sort=False,
            dropna=False,
        )

    for group_name, group_df in grouped:
        group_df = group_df.copy()

        if time_column is not None:
            group_df = group_df.sort_values(
                time_column
            ).reset_index(drop=True)

            new_segment = (
                group_df[time_column]
                .diff()
                .ne(pd.Timedelta(hours=1))
            )
            group_df["_sequence_segment"] = new_segment.cumsum()
            segments = group_df.groupby(
                "_sequence_segment",
                sort=False,
            )
        else:
            segments = [(0, group_df.reset_index(drop=True))]

        for segment_id, segment_df in segments:
            yield (
                str(group_name),
                int(segment_id),
                segment_df.reset_index(drop=True),
                time_column,
            )


def create_sequences(
    dataframe: pd.DataFrame,
    split_name: str,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[SequenceMetadata]]:
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    metadata: list[SequenceMetadata] = []
    minimum_rows = INPUT_HOURS + OUTPUT_HOURS

    for (
        group_name,
        segment_id,
        segment_df,
        time_column,
    ) in split_into_contiguous_groups(dataframe, split_name):
        if len(segment_df) < minimum_rows:
            continue

        features = segment_df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )
        target = segment_df[TARGET_COLUMN].to_numpy(
            dtype=np.float32
        )
        last_start = len(segment_df) - minimum_rows + 1

        for start_index in range(0, last_start, stride):
            input_end = start_index + INPUT_HOURS
            target_end = input_end + OUTPUT_HOURS

            X.append(features[start_index:input_end])
            y.append(target[input_end:target_end])

            forecast_start = None
            if time_column is not None:
                forecast_start = (
                    segment_df.loc[input_end, time_column].isoformat()
                )

            metadata.append(
                SequenceMetadata(
                    group=group_name,
                    segment=segment_id,
                    start_index=start_index,
                    forecast_start=forecast_start,
                )
            )

    if not X:
        raise ValueError(
            f"No {split_name} sequences were created. Each uninterrupted "
            f"location segment needs at least {minimum_rows} hourly rows."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        metadata,
    )


def make_event_sample_weights(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Optionally emphasize forecasts with meaningful temperature transitions.

    The weight uses both the sharpest hourly change and the full 24-hour
    net change. Keep this optional and compare it through an ablation run.
    """
    last_observed = X[:, -1, TEMPERATURE_FEATURE_INDEX]
    path = np.concatenate(
        [last_observed[:, np.newaxis], y],
        axis=1,
    )

    largest_hourly_change = np.max(
        np.abs(np.diff(path, axis=1)),
        axis=1,
    )
    net_change = np.abs(y[:, -1] - last_observed)

    sharpness_strength = np.clip(
        largest_hourly_change / 4.0,
        0.0,
        1.0,
    )
    transition_strength = np.clip(
        net_change / 8.0,
        0.0,
        1.0,
    )

    return (
        1.0
        + 0.20 * sharpness_strength
        + 0.20 * transition_strength
    ).astype(np.float32)


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    errors = predicted - actual
    absolute_errors = np.abs(errors)
    per_sample_mae = np.mean(absolute_errors, axis=1)

    return {
        "overall_mae": float(np.mean(absolute_errors)),
        "overall_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "overall_bias": float(np.mean(errors)),
        "max_absolute_error": float(np.max(absolute_errors)),
        "median_sample_mae": float(np.median(per_sample_mae)),
        "p90_sample_mae": float(np.percentile(per_sample_mae, 90)),
        "p95_sample_mae": float(np.percentile(per_sample_mae, 95)),
        "p99_sample_mae": float(np.percentile(per_sample_mae, 99)),
        "endpoint_mae": float(np.mean(absolute_errors[:, -1])),
        "mae_by_hour": np.mean(
            absolute_errors,
            axis=0,
        ).astype(float).tolist(),
        "rmse_by_hour": np.sqrt(
            np.mean(np.square(errors), axis=0)
        ).astype(float).tolist(),
        "bias_by_hour": np.mean(
            errors,
            axis=0,
        ).astype(float).tolist(),
    }


def calculate_group_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metadata: list[SequenceMetadata],
) -> dict[str, dict]:
    groups = np.asarray([item.group for item in metadata], dtype=object)
    results: dict[str, dict] = {}

    for group in sorted(set(groups.tolist())):
        mask = groups == group
        results[group] = calculate_metrics(
            actual[mask],
            predicted[mask],
        )
        results[group]["samples"] = int(np.sum(mask))

    return results


def save_prediction_table(
    path: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    metadata: list[SequenceMetadata],
) -> None:
    records: list[dict] = []

    for sample_index, sample_meta in enumerate(metadata):
        for horizon_index in range(OUTPUT_HOURS):
            records.append(
                {
                    "sample": sample_index,
                    "forecast_hour": horizon_index + 1,
                    "group": sample_meta.group,
                    "segment": sample_meta.segment,
                    "forecast_start": sample_meta.forecast_start,
                    "actual": float(actual[sample_index, horizon_index]),
                    "predicted": float(
                        predicted[sample_index, horizon_index]
                    ),
                    "error": float(
                        predicted[sample_index, horizon_index]
                        - actual[sample_index, horizon_index]
                    ),
                }
            )

    pd.DataFrame(records).to_csv(path, index=False)


def save_plots(
    figure_dir: Path,
    history: tf.keras.callbacks.History,
    actual: np.ndarray,
    predicted: np.ndarray,
    forecast_metrics: dict,
    previous_day_metrics: dict,
) -> int:
    figure_dir.mkdir(parents=True, exist_ok=True)

    history_dict = history.history
    val_mae = np.asarray(history_dict["val_mae"], dtype=float)
    best_epoch = int(np.argmin(val_mae))

    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch: {best_epoch + 1}",
    )
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Level + Change Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        figure_dir / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["mae"], label="Training MAE")
    plt.plot(history_dict["val_mae"], label="Validation MAE")
    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch: {best_epoch + 1}",
    )
    plt.title("Training vs Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        figure_dir / "mae_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Use the sample nearest the median test MAE rather than always sample 0.
    per_sample_mae = np.mean(
        np.abs(predicted - actual),
        axis=1,
    )
    median_mae = float(np.median(per_sample_mae))
    representative_index = int(
        np.argmin(np.abs(per_sample_mae - median_mae))
    )

    forecast_hours = np.arange(1, OUTPUT_HOURS + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(
        forecast_hours,
        actual[representative_index],
        marker="o",
        label="Actual Temperature",
    )
    plt.plot(
        forecast_hours,
        predicted[representative_index],
        marker="o",
        label="Predicted Temperature",
    )
    plt.title(
        "Representative 24-Hour Temperature Forecast "
        f"(sample MAE {per_sample_mae[representative_index]:.2f} °C)"
    )
    plt.xlabel("Forecast Hour")
    plt.ylabel("Temperature (°C)")
    plt.xticks(forecast_hours)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        figure_dir / "forecast_24h_example.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        forecast_hours,
        forecast_metrics["mae_by_hour"],
        marker="o",
        label="Improved Model MAE",
    )
    plt.plot(
        forecast_hours,
        previous_day_metrics["mae_by_hour"],
        marker="o",
        label="Previous-Day Baseline MAE",
    )
    plt.title("MAE by Forecast Horizon")
    plt.xlabel("Forecast Hour")
    plt.ylabel("Mean Absolute Error (°C)")
    plt.xticks(forecast_hours)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        figure_dir / "mae_by_forecast_hour.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return representative_index


def parse_args() -> argparse.Namespace:
    project_root = discover_project_root()
    split_dir = project_root / "data" / "splits"

    parser = argparse.ArgumentParser(
        description="Train the regularized 72-to-24-hour LSTM model."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=split_dir / "train.csv",
    )
    parser.add_argument(
        "--dev-csv",
        type=Path,
        default=split_dir / "dev.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=split_dir / "test.csv",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help=(
            "Window step. A value of 3 reduces near-duplicate overlapping "
            "samples and usually generalizes better than stride 1."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--event-weighting",
        action="store_true",
        help=(
            "Opt in to mild extra weighting for sharp or sustained "
            "temperature transitions. Leave off for the clean baseline run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(RANDOM_SEED)

    for required_file in [
        args.train_csv,
        args.dev_csv,
        args.test_csv,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {required_file}"
            )

    project_root = discover_project_root()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_root = project_root / "models" / "LSTM"
    experiment_dir = model_root / "experiments" / timestamp
    figure_dir = experiment_dir / "figures"
    latest_dir = model_root / "latest"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = experiment_dir / "weather_lstm.keras"
    latest_model_path = latest_dir / "weather_lstm.keras"
    config_path = experiment_dir / "model_config.json"
    latest_config_path = latest_dir / "model_config.json"
    metrics_path = experiment_dir / "metrics.json"
    predictions_path = experiment_dir / "predictions.csv"
    history_path = experiment_dir / "training_history.csv"
    summary_path = experiment_dir / "model_summary.txt"

    print("Loading datasets...")
    train_df = validate_dataframe(
        pd.read_csv(args.train_csv),
        "training data",
    )
    dev_df = validate_dataframe(
        pd.read_csv(args.dev_csv),
        "development data",
    )
    test_df = validate_dataframe(
        pd.read_csv(args.test_csv),
        "testing data",
    )

    split_ranges = audit_chronological_splits(
        train_df,
        dev_df,
        test_df,
    )

    print("Creating contiguous forecast sequences...")
    X_train, y_train, train_metadata = create_sequences(
        train_df,
        "training data",
        stride=args.stride,
    )
    X_dev, y_dev, dev_metadata = create_sequences(
        dev_df,
        "development data",
        stride=args.stride,
    )
    X_test, y_test, test_metadata = create_sequences(
        test_df,
        "testing data",
        stride=args.stride,
    )

    print(f"Training samples    : {len(X_train)}")
    print(f"Development samples : {len(X_dev)}")
    print(f"Testing samples     : {len(X_test)}")
    print(f"Input shape         : {X_train.shape}")
    print(f"Target shape        : {y_train.shape}")
    print(f"Window stride       : {args.stride}")

    train_sample_weights = None
    dev_sample_weights = None

    if args.event_weighting:
        train_sample_weights = make_event_sample_weights(
            X_train,
            y_train,
        )
        dev_sample_weights = make_event_sample_weights(
            X_dev,
            y_dev,
        )
        print(
            "Training event weights: "
            f"mean={train_sample_weights.mean():.3f}, "
            f"min={train_sample_weights.min():.3f}, "
            f"max={train_sample_weights.max():.3f}"
        )
        print(
            "Development event weights: "
            f"mean={dev_sample_weights.mean():.3f}, "
            f"min={dev_sample_weights.min():.3f}, "
            f"max={dev_sample_weights.max():.3f}"
        )

    print("\nBuilding cross-attentive multi-baseline LSTM...")
    model = build_weather_model(
        X_train,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model.summary()

    with summary_path.open("w", encoding="utf-8") as file:
        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            min_delta=0.002,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            mode="min",
            factor=0.5,
            patience=4,
            min_delta=0.001,
            cooldown=1,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=history_path,
            append=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("\nBeginning training...\n")
    history = model.fit(
        X_train,
        y_train,
        sample_weight=train_sample_weights,
        validation_data=(
            (X_dev, y_dev, dev_sample_weights)
            if dev_sample_weights is not None
            else (X_dev, y_dev)
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Always evaluate the checkpoint with the lowest unweighted validation MAE.
    best_model = load_weather_model(
        best_model_path,
        compile=True,
    )

    # These unweighted evaluations make train/dev/test MAE directly comparable,
    # even when optional event weighting was used during optimization.
    train_evaluation = best_model.evaluate(
        X_train,
        y_train,
        verbose=0,
        return_dict=True,
    )
    dev_evaluation = best_model.evaluate(
        X_dev,
        y_dev,
        verbose=0,
        return_dict=True,
    )
    dev_predictions = np.asarray(
        best_model.predict(X_dev, verbose=0),
        dtype=np.float32,
    )
    dev_forecast_metrics = calculate_metrics(
        y_dev,
        dev_predictions,
    )
    dev_group_metrics = calculate_group_metrics(
        y_dev,
        dev_predictions,
        dev_metadata,
    )

    evaluation = best_model.evaluate(
        X_test,
        y_test,
        verbose=0,
        return_dict=True,
    )
    predictions = np.asarray(
        best_model.predict(X_test, verbose=0),
        dtype=np.float32,
    )

    forecast_metrics = calculate_metrics(
        y_test,
        predictions,
    )

    last_observed = X_test[
        :,
        -1,
        TEMPERATURE_FEATURE_INDEX,
    ]
    persistence_predictions = np.repeat(
        last_observed[:, np.newaxis],
        OUTPUT_HOURS,
        axis=1,
    )
    previous_day_predictions = X_test[
        :,
        -OUTPUT_HOURS:,
        TEMPERATURE_FEATURE_INDEX,
    ]

    persistence_metrics = calculate_metrics(
        y_test,
        persistence_predictions,
    )
    previous_day_metrics = calculate_metrics(
        y_test,
        previous_day_predictions,
    )
    group_metrics = calculate_group_metrics(
        y_test,
        predictions,
        test_metadata,
    )

    improvement_percent = float(
        100.0
        * (
            previous_day_metrics["overall_mae"]
            - forecast_metrics["overall_mae"]
        )
        / previous_day_metrics["overall_mae"]
    )

    representative_index = save_plots(
        figure_dir,
        history,
        y_test,
        predictions,
        forecast_metrics,
        previous_day_metrics,
    )
    save_prediction_table(
        predictions_path,
        y_test,
        predictions,
        test_metadata,
    )

    best_epoch = int(
        np.argmin(history.history["val_mae"])
    ) + 1

    metrics = {
        "best_epoch": best_epoch,
        "unweighted_evaluation": {
            "train": {
                key: float(value)
                for key, value in train_evaluation.items()
            },
            "dev": {
                key: float(value)
                for key, value in dev_evaluation.items()
            },
            "test": {
                key: float(value)
                for key, value in evaluation.items()
            },
        },
        "development_model": dev_forecast_metrics,
        "development_metrics_by_group": dev_group_metrics,
        "improved_model": forecast_metrics,
        "test_metrics_by_group": group_metrics,
        "persistence_baseline": persistence_metrics,
        "previous_day_baseline": previous_day_metrics,
        "improvement_vs_previous_day_mae_percent": improvement_percent,
        "representative_plot_sample": representative_index,
        "training": {
            "stride": args.stride,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history.history["loss"]),
            "batch_size": args.batch_size,
            "event_weighting": args.event_weighting,
            "selection_metric": "val_mae",
            "random_seed": RANDOM_SEED,
        },
        "split_ranges": split_ranges,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Publish exactly the best checkpoint and its matching configuration.
    shutil.copy2(best_model_path, latest_model_path)

    config = {
        "model_name": best_model.name,
        "input_hours": INPUT_HOURS,
        "output_hours": OUTPUT_HOURS,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "temperature_feature_index": TEMPERATURE_FEATURE_INDEX,
        "created_at": timestamp,
        "best_epoch": best_epoch,
        "training_stride": args.stride,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "event_weighting": args.event_weighting,
        "selection_metric": "val_mae",
        "random_seed": RANDOM_SEED,
        "training_files": {
            "train": str(args.train_csv.resolve()),
            "dev": str(args.dev_csv.resolve()),
            "test": str(args.test_csv.resolve()),
        },
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)
    shutil.copy2(config_path, latest_config_path)

    print("\nTest results")
    print("-" * 64)
    print(f"Best validation epoch : {best_epoch}")
    print(
        f"Train MAE (unweighted): "
        f"{train_evaluation['mae']:.4f} °C"
    )
    print(
        f"Dev MAE (unweighted)  : "
        f"{dev_evaluation['mae']:.4f} °C"
    )
    print(
        f"Dev p90 sample MAE    : "
        f"{dev_forecast_metrics['p90_sample_mae']:.4f} °C"
    )
    print(
        f"Dev hour-24 MAE       : "
        f"{dev_forecast_metrics['endpoint_mae']:.4f} °C"
    )
    print(
        f"Test MAE              : "
        f"{forecast_metrics['overall_mae']:.4f} °C"
    )
    print(
        f"Test RMSE             : "
        f"{forecast_metrics['overall_rmse']:.4f} °C"
    )
    print(
        f"Overall bias          : "
        f"{forecast_metrics['overall_bias']:+.4f} °C"
    )
    print(
        f"Median sample MAE     : "
        f"{forecast_metrics['median_sample_mae']:.4f} °C"
    )
    print(
        f"90th-percentile MAE   : "
        f"{forecast_metrics['p90_sample_mae']:.4f} °C"
    )
    print(
        f"95th-percentile MAE   : "
        f"{forecast_metrics['p95_sample_mae']:.4f} °C"
    )
    print(
        f"Hour-24 endpoint MAE  : "
        f"{forecast_metrics['endpoint_mae']:.4f} °C"
    )
    print(
        f"Persistence MAE       : "
        f"{persistence_metrics['overall_mae']:.4f} °C"
    )
    print(
        f"Previous-day MAE      : "
        f"{previous_day_metrics['overall_mae']:.4f} °C"
    )
    print(
        "MAE improvement vs previous day: "
        f"{improvement_percent:.2f}%"
    )

    print("\nMAE by forecast hour")
    print("-" * 64)
    for hour, value in enumerate(
        forecast_metrics["mae_by_hour"],
        start=1,
    ):
        print(f"Hour {hour:2d}: {value:.3f} °C")

    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"Latest model saved to: {latest_model_path}")


if __name__ == "__main__":
    main()