"""Generates a 24-hour forecast from one experiment's latest saved model.

Usage:
    python -m src.models.transformer.predict --experiment exp03_temperature_city --city seattle
    python -m src.models.transformer.predict --experiment exp00_baseline --city boston --input-csv data/splits/test.csv
"""

import argparse
import importlib

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.common.data import (
    CITY_COLUMN,
    CITY_VOCAB,
    INPUT_HOURS,
    OUTPUT_HOURS,
    PROJECT_ROOT,
    TIME_COLUMN,
    load_splits,
    split_into_contiguous_groups,
)

MODELS_DIR = PROJECT_ROOT / "models" / "Transformer"
EXPERIMENTS_PACKAGE = "src.models.transformer.experiments"


def latest_experiment_dir(experiment_name):
    experiments_dir = MODELS_DIR / experiment_name / "experiments"
    if not experiments_dir.exists():
        raise FileNotFoundError(
            f"No experiments found for '{experiment_name}' under {experiments_dir}. "
            f"Available experiments: {', '.join(sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir()))}"
        )

    candidates = sorted((d for d in experiments_dir.iterdir() if d.is_dir()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No trained runs found under {experiments_dir}.")

    return candidates[0]


def load_feature_columns(experiment_name):
    """Imports the experiment's own config.py rather than hardcoding a
    feature list here, so predict.py always matches whatever that
    experiment was actually trained on."""
    config = importlib.import_module(f"{EXPERIMENTS_PACKAGE}.{experiment_name}.config")
    return config.FEATURE_COLUMNS


def latest_contiguous_window(dataframe, feature_columns, input_hours):
    """Finds this city's most recent uninterrupted segment with at least
    input_hours rows and returns its last input_hours rows as (input_hours, F)
    plus the timestamp of the final observed hour."""
    best_segment = None
    for _, segment_df in split_into_contiguous_groups(dataframe):
        if len(segment_df) < input_hours:
            continue
        if best_segment is None or segment_df[TIME_COLUMN].iloc[-1] > best_segment[TIME_COLUMN].iloc[-1]:
            best_segment = segment_df

    if best_segment is None:
        raise ValueError(
            f"No uninterrupted {input_hours}-hour segment was found for this city."
        )

    window = best_segment.iloc[-input_hours:]
    features = window[feature_columns].to_numpy(dtype=np.float32)
    last_timestamp = pd.Timestamp(window[TIME_COLUMN].iloc[-1])
    return features, last_timestamp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 24-hour forecast from one experiment's latest saved model."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment folder name under models/Transformer/, e.g. exp03_temperature_city."
    )
    parser.add_argument(
        "--city", required=True,
        help=f"City name. Available: {', '.join(sorted(CITY_VOCAB))}"
    )
    parser.add_argument(
        "--input-csv", type=str, default=None,
        help="Optional CSV with time/city/feature columns to read the input window from. "
             "Defaults to the shared test split (data/splits/test.csv)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    city = args.city.strip().lower()
    if city not in CITY_VOCAB:
        raise ValueError(f"Unknown city '{args.city}'. Available cities: {', '.join(sorted(CITY_VOCAB))}")

    experiment_dir = latest_experiment_dir(args.experiment)

    # Import the experiment's config (and therefore training.py, which sets
    # TF's intra-op thread count at import time) before any TF op runs --
    # that setting can't be changed once the TF context has initialized.
    feature_columns = load_feature_columns(args.experiment)

    model_path = experiment_dir / "weather_transformer.keras"
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if args.input_csv is not None:
        print(f"Loading data: {args.input_csv}")
        dataframe = pd.read_csv(args.input_csv)
    else:
        print("Loading data: shared test split")
        _, _, dataframe = load_splits()

    city_rows = dataframe[dataframe[CITY_COLUMN].str.lower() == city]
    if city_rows.empty:
        raise ValueError(f"No rows found for city '{city}' in the input data.")

    features, last_timestamp = latest_contiguous_window(city_rows, feature_columns, INPUT_HOURS)
    X = features[np.newaxis, ...]

    city_aware = len(model.inputs) == 2
    if city_aware:
        city_id = np.array([CITY_VOCAB[city]], dtype=np.int32)
        model_input = [X, city_id]
    else:
        model_input = X

    predictions = np.asarray(model.predict(model_input, verbose=0), dtype=float)[0]

    forecast_start = last_timestamp.floor("h") + pd.Timedelta(hours=1)
    forecast_times = pd.date_range(start=forecast_start, periods=OUTPUT_HOURS, freq="h")

    forecast_df = pd.DataFrame({
        "forecast_hour": np.arange(1, OUTPUT_HOURS + 1),
        "forecast_time": forecast_times,
        "predicted_temperature_c": predictions,
    })

    print(f"\n{args.experiment} forecast for {city.title()} ({'city-aware' if city_aware else 'single-input'} model)")
    print(f"Last observation : {last_timestamp}")
    print(f"Forecast starts  : {forecast_start}")
    print("-" * 60)
    print(forecast_df.to_string(
        index=False,
        formatters={"predicted_temperature_c": lambda value: f"{value:.2f} °C"},
    ))

    output_dir = MODELS_DIR / args.experiment / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{city.replace(' ', '_')}_forecast.csv"
    forecast_df.to_csv(output_path, index=False)
    print(f"\nForecast saved to: {output_path}")


if __name__ == "__main__":
    main()
