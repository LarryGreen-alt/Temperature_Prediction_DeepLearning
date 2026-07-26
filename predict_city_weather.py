from __future__ import annotations

from datetime import datetime
from math import ceil
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================


def find_project_root() -> Path:
    """
    Locate the project root so this script works from either the
    project root or a subfolder such as src/predictions/.
    """
    script_directory = Path(__file__).resolve().parent
    candidates = [script_directory, *script_directory.parents]

    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "src").exists():
            return candidate

    return script_directory


PROJECT_ROOT = find_project_root()

project_root_string = str(PROJECT_ROOT)
if project_root_string not in sys.path:
    sys.path.insert(0, project_root_string)

from src.utils.city_coordinates import CITY_COORDINATES


MODELS_DIRECTORY = PROJECT_ROOT / "models"
EXPERIMENT_DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

# The revised LSTM can return hours 1 through 24 in one model call.
MAX_FORECAST_HOURS = 24

TEMPERATURE_CELSIUS = "°C"
TEMPERATURE_FAHRENHEIT = "°F"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# Older models used ten inputs. The revised LSTM uses eleven because
# historical temperature_2m is now included as an input.
LEGACY_FEATURE_COLUMNS = [
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

REVISED_LSTM_FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

# Columns that Open-Meteo can provide directly. The cyclical time columns
# are calculated locally after download.
OPEN_METEO_FEATURE_COLUMNS = {
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
}


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lstm": {
        "display_name": "LSTM",
        "model_root": MODELS_DIRECTORY / "LSTM",
        "experiments_directory": MODELS_DIRECTORY / "LSTM" / "experiments",
        "latest_directory": MODELS_DIRECTORY / "LSTM" / "latest",
        "model_filename": "weather_lstm.keras",
        "fallback_feature_columns": REVISED_LSTM_FEATURE_COLUMNS,
        "fallback_forecast_horizon": 1,
    },
    "transformer": {
        "display_name": "Transformer",
        "model_root": MODELS_DIRECTORY / "Transformer",
        "experiments_directory": MODELS_DIRECTORY / "Transformer" / "experiments",
        "latest_directory": MODELS_DIRECTORY / "Transformer" / "latest",
        "model_filename": "weather_transformer.keras",
        "fallback_feature_columns": LEGACY_FEATURE_COLUMNS,
        "fallback_forecast_horizon": 1,
    },
}


# ============================================================
# USER SELECTION
# ============================================================


def select_model_type() -> str:
    print("\nChoose a prediction model")
    print("=" * 40)
    print("1. LSTM")
    print("2. Transformer")

    while True:
        choice = input("\nEnter 1 or 2: ").strip().lower()

        if choice in {"1", "lstm"}:
            return "lstm"

        if choice in {"2", "transformer"}:
            return "transformer"

        print("Invalid selection. Enter 1 for LSTM or 2 for Transformer.")


def normalize_city_name(city: str) -> str:
    return " ".join(
        city.strip().lower().replace("_", " ").replace(",", " ").split()
    )


def select_city() -> str:
    if not CITY_COORDINATES:
        raise ValueError("CITY_COORDINATES is empty.")

    cities = sorted(CITY_COORDINATES)

    print("\nAvailable cities")
    print("=" * 50)

    for number, city in enumerate(cities, start=1):
        print(f"{number:>2}. {city.title()}")

    while True:
        choice = input("\nEnter a city name or its number: ").strip()

        if choice.isdigit():
            city_number = int(choice)

            if 1 <= city_number <= len(cities):
                return cities[city_number - 1]

        normalized_city = normalize_city_name(choice)

        if normalized_city in CITY_COORDINATES:
            return normalized_city

        print("City not found. Please choose a city from the displayed list.")


def select_forecast_hours() -> int:
    while True:
        raw_value = input(
            f"\nHow many hours ahead should be predicted (1-{MAX_FORECAST_HOURS})? "
        ).strip()

        try:
            hours = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if hours < 1:
            print("The minimum forecast is 1 hour.")
            continue

        if hours > MAX_FORECAST_HOURS:
            print(
                f"The requested forecast exceeds the limit. "
                f"Using {MAX_FORECAST_HOURS} hours."
            )
            return MAX_FORECAST_HOURS

        return hours


# ============================================================
# MODEL DISCOVERY, CONFIGURATION, AND LOADING
# ============================================================


def parse_experiment_datetime(
    experiment_directory: Path,
) -> datetime | None:
    try:
        return datetime.strptime(
            experiment_directory.name,
            EXPERIMENT_DATE_FORMAT,
        )
    except ValueError:
        return None


def find_latest_experiment_model(
    model_type: str,
) -> tuple[Path, Path]:
    """
    Prefer models/<type>/latest because train.py copies the completed,
    best model there. Fall back to the newest timestamped experiment.
    """
    model_type = model_type.strip().lower()

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    config = MODEL_CONFIGS[model_type]
    expected_filename = str(config["model_filename"])
    latest_directory = Path(config["latest_directory"])
    latest_model_path = latest_directory / expected_filename

    if latest_model_path.is_file():
        return latest_model_path, latest_directory

    experiments_directory = Path(config["experiments_directory"])

    if not experiments_directory.exists():
        raise FileNotFoundError(
            f"The {config['display_name']} experiments directory "
            f"was not found:\n{experiments_directory}"
        )

    timestamped_models: list[tuple[datetime, Path, Path]] = []
    fallback_models: list[Path] = []

    for experiment_directory in experiments_directory.iterdir():
        if not experiment_directory.is_dir():
            continue

        expected_model_path = experiment_directory / expected_filename

        if expected_model_path.is_file():
            keras_files = [expected_model_path]
        else:
            keras_files = sorted(experiment_directory.glob("*.keras"))

        experiment_datetime = parse_experiment_datetime(experiment_directory)

        for keras_file in keras_files:
            if experiment_datetime is not None:
                timestamped_models.append(
                    (
                        experiment_datetime,
                        experiment_directory,
                        keras_file,
                    )
                )
            else:
                fallback_models.append(keras_file)

    if timestamped_models:
        _, experiment_directory, model_path = max(
            timestamped_models,
            key=lambda item: item[0],
        )
        return model_path, experiment_directory

    if fallback_models:
        model_path = max(
            fallback_models,
            key=lambda path: path.stat().st_mtime,
        )
        return model_path, model_path.parent

    raise FileNotFoundError(
        f"No usable .keras model was found inside:\n"
        f"{experiments_directory}\n\n"
        f"Expected filename: {expected_filename}"
    )


def load_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Model configuration could not be read:\n{config_path}\n{error}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"Model configuration must contain a JSON object:\n{config_path}"
        )

    return value


def get_model_dimensions(model: tf.keras.Model) -> tuple[int, int]:
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                "This script supports one model input, but the selected model "
                f"has {len(input_shape)} inputs."
            )
        input_shape = input_shape[0]

    if len(input_shape) != 3:
        raise ValueError(
            "Expected a sequence model input shaped "
            "(batch, sequence_length, feature_count), but received "
            f"{input_shape}."
        )

    sequence_length = input_shape[1]
    feature_count = input_shape[2]

    if sequence_length is None or feature_count is None:
        raise ValueError(
            "The model input shape must have fixed sequence and feature "
            f"dimensions. Received: {input_shape}"
        )

    return int(sequence_length), int(feature_count)


def get_model_output_count(model: tf.keras.Model) -> int:
    output_shape = model.output_shape

    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            raise ValueError(
                "This script supports one model output, but the selected model "
                f"has {len(output_shape)} outputs."
            )
        output_shape = output_shape[0]

    if len(output_shape) == 1:
        return 1

    output_count = output_shape[-1]

    if output_count is None:
        raise ValueError(
            f"The model output dimension must be fixed. Received: {output_shape}"
        )

    return int(output_count)


def resolve_feature_columns(
    model_type: str,
    model: tf.keras.Model,
    saved_config: dict[str, Any],
) -> list[str]:
    """
    Use the exact feature order saved by train.py. If an older model has no
    config file, select a safe fallback based on the model's input width.
    """
    _, expected_feature_count = get_model_dimensions(model)

    configured_features = saved_config.get("feature_columns")

    if isinstance(configured_features, list) and all(
        isinstance(column, str) for column in configured_features
    ):
        feature_columns = configured_features
    else:
        fallback_features = list(
            MODEL_CONFIGS[model_type]["fallback_feature_columns"]
        )

        if expected_feature_count == len(fallback_features):
            feature_columns = fallback_features
        elif expected_feature_count == len(LEGACY_FEATURE_COLUMNS):
            feature_columns = list(LEGACY_FEATURE_COLUMNS)
        elif expected_feature_count == len(REVISED_LSTM_FEATURE_COLUMNS):
            feature_columns = list(REVISED_LSTM_FEATURE_COLUMNS)
        else:
            raise ValueError(
                "The model has no usable model_config.json and its feature "
                f"count ({expected_feature_count}) does not match a known "
                "feature layout."
            )

    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            "Model feature mismatch.\n"
            f"The selected model expects {expected_feature_count} features, "
            f"but its resolved configuration supplies {len(feature_columns)}.\n"
            f"Resolved feature order: {feature_columns}"
        )

    return feature_columns


def load_selected_model(
    model_type: str,
) -> tuple[tf.keras.Model, Path, Path, dict[str, Any], list[str]]:
    model_path, model_directory = find_latest_experiment_model(model_type)
    display_name = str(MODEL_CONFIGS[model_type]["display_name"])

    config_candidates = [
        model_directory / "model_config.json",
        model_path.with_name("model_config.json"),
        Path(MODEL_CONFIGS[model_type]["latest_directory"]) / "model_config.json",
    ]

    saved_config: dict[str, Any] = {}
    config_path_used: Path | None = None

    for config_path in config_candidates:
        if config_path.is_file():
            saved_config = load_json_config(config_path)
            config_path_used = config_path
            break

    print(f"\nSelected {display_name} model:")
    print(f"Directory: {model_directory}")
    print(f"Model:     {model_path}")

    if config_path_used is not None:
        print(f"Config:    {config_path_used}")
    else:
        print("Config:    no model_config.json found; using inferred settings")

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"The selected {display_name} model could not be loaded.\n\n"
            f"Directory: {model_directory}\n"
            f"Model: {model_path}\n"
            f"Original error: {error}"
        ) from error

    feature_columns = resolve_feature_columns(
        model_type=model_type,
        model=model,
        saved_config=saved_config,
    )

    return model, model_path, model_directory, saved_config, feature_columns


# ============================================================
# WEATHER INPUT DATA
# ============================================================


def get_api_feature_columns(feature_columns: list[str]) -> list[str]:
    generated_time_features = {
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    }

    unsupported_columns = [
        column
        for column in feature_columns
        if column not in OPEN_METEO_FEATURE_COLUMNS
        and column not in generated_time_features
    ]

    if unsupported_columns:
        raise ValueError(
            "The selected model requires features this city forecast script "
            f"does not know how to create: {unsupported_columns}"
        )

    return [
        column
        for column in feature_columns
        if column in OPEN_METEO_FEATURE_COLUMNS
    ]


def download_weather_features(
    latitude: float,
    longitude: float,
    required_history_hours: int,
    api_feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    """
    Download enough recent hourly data to build the model's input window.
    """
    past_days = max(2, ceil((required_history_hours + 24) / 24))

    if past_days > 92:
        raise ValueError(
            f"The model requires {required_history_hours} historical hours, "
            "which exceeds this script's 92-day retrieval limit."
        )

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(api_feature_columns),
        "past_days": past_days,
        "forecast_days": 2,
        "timezone": "auto",
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=parameters,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Weather data could not be downloaded: {error}"
        ) from error

    payload = response.json()

    if "hourly" not in payload:
        reason = payload.get(
            "reason",
            "The weather service did not return hourly data.",
        )
        raise RuntimeError(f"Weather service error: {reason}")

    weather = pd.DataFrame(payload["hourly"])

    required_api_columns = ["time", *api_feature_columns]
    missing_columns = [
        column
        for column in required_api_columns
        if column not in weather.columns
    ]

    if missing_columns:
        raise ValueError(
            "The weather response is missing required columns: "
            f"{missing_columns}"
        )

    weather["time"] = pd.to_datetime(
        weather["time"],
        errors="coerce",
    )

    weather = (
        weather.dropna(subset=["time"])
        .drop_duplicates(subset=["time"], keep="last")
        .set_index("time")
        .sort_index()
    )

    timezone_name = str(payload.get("timezone", "UTC"))

    try:
        local_now = (
            pd.Timestamp.now(tz=timezone_name)
            .floor("h")
            .tz_localize(None)
        )
    except Exception:
        utc_offset_seconds = int(payload.get("utc_offset_seconds", 0))
        local_now = (
            pd.Timestamp.now(tz="UTC")
            + pd.Timedelta(seconds=utc_offset_seconds)
        ).floor("h").tz_localize(None)

    return weather, local_now, timezone_name


def add_time_features(
    weather: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Create the same cyclical time features used during training and arrange
    every model feature in the exact saved order.
    """
    result = weather.copy()

    hour = result.index.hour
    day_of_year = result.index.dayofyear

    if "hour_sin" in feature_columns:
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)

    if "hour_cos" in feature_columns:
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    if "day_sin" in feature_columns:
        result["day_sin"] = np.sin(
            2 * np.pi * day_of_year / 365.25
        )

    if "day_cos" in feature_columns:
        result["day_cos"] = np.cos(
            2 * np.pi * day_of_year / 365.25
        )

    missing_columns = [
        column for column in feature_columns if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Weather data is missing required model features: {missing_columns}"
        )

    for column in feature_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result[feature_columns] = result[feature_columns].ffill().bfill()

    if result[feature_columns].isna().any().any():
        bad_columns = result[feature_columns].columns[
            result[feature_columns].isna().any()
        ].tolist()

        raise ValueError(
            "Some model features still contain missing values after cleanup: "
            f"{bad_columns}"
        )

    return result


# ============================================================
# PREDICTION
# ============================================================


def build_input_window(
    weather: pd.DataFrame,
    feature_columns: list[str],
    window_end_time: pd.Timestamp,
    sequence_length: int,
) -> np.ndarray:
    input_window = weather.loc[
        weather.index <= window_end_time,
        feature_columns,
    ].tail(sequence_length)

    if len(input_window) < sequence_length:
        raise ValueError(
            f"The model requires {sequence_length} hourly rows, but only "
            f"{len(input_window)} were available before {window_end_time}."
        )

    return np.expand_dims(
        input_window.to_numpy(dtype=np.float32),
        axis=0,
    )


def predict_multi_output_model(
    model: tf.keras.Model,
    weather: pd.DataFrame,
    feature_columns: list[str],
    current_hour: pd.Timestamp,
    requested_hours: int,
    sequence_length: int,
    output_count: int,
) -> pd.DataFrame:
    """
    Revised LSTM path: one 72-hour window produces a vector containing
    forecast hours 1 through 24.
    """
    if requested_hours > output_count:
        raise ValueError(
            f"The selected model returns {output_count} forecast hours, "
            f"but {requested_hours} were requested."
        )

    model_input = build_input_window(
        weather=weather,
        feature_columns=feature_columns,
        window_end_time=current_hour,
        sequence_length=sequence_length,
    )

    raw_prediction = model.predict(model_input, verbose=0)
    prediction_values = np.asarray(raw_prediction).reshape(-1)

    if prediction_values.size != output_count:
        raise ValueError(
            f"The model output shape was expected to contain {output_count} "
            f"values, but received {np.asarray(raw_prediction).shape}."
        )

    records = []

    for horizon_index in range(requested_hours):
        predicted_temperature = float(prediction_values[horizon_index])

        if not np.isfinite(predicted_temperature):
            raise ValueError(
                "The model returned an invalid temperature at forecast hour "
                f"{horizon_index + 1}: {predicted_temperature}"
            )

        records.append(
            {
                "hours_ahead": horizon_index + 1,
                "forecast_time": (
                    current_hour + pd.Timedelta(hours=horizon_index + 1)
                ),
                "predicted_temperature": predicted_temperature,
            }
        )

    return pd.DataFrame(records)


def predict_single_output_model(
    model: tf.keras.Model,
    model_type: str,
    weather: pd.DataFrame,
    feature_columns: list[str],
    current_hour: pd.Timestamp,
    requested_hours: int,
    sequence_length: int,
    saved_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Legacy model path: make one model call per requested target hour.
    This remains useful for the existing single-output Transformer.
    """
    forecast_horizon = int(
        saved_config.get(
            "forecast_horizon",
            MODEL_CONFIGS[model_type]["fallback_forecast_horizon"],
        )
    )

    if forecast_horizon < 1:
        raise ValueError(
            "The configured model forecast horizon must be at least 1."
        )

    if requested_hours < forecast_horizon:
        raise ValueError(
            f"The {MODEL_CONFIGS[model_type]['display_name']} model was trained "
            f"for a {forecast_horizon}-hour horizon, so it cannot produce a "
            f"{requested_hours}-hour-ahead prediction with this configuration."
        )

    predictions: list[dict[str, Any]] = []

    for hour_ahead in range(forecast_horizon, requested_hours + 1):
        target_time = current_hour + pd.Timedelta(hours=hour_ahead)
        window_end_time = target_time - pd.Timedelta(hours=forecast_horizon)

        model_input = build_input_window(
            weather=weather,
            feature_columns=feature_columns,
            window_end_time=window_end_time,
            sequence_length=sequence_length,
        )

        raw_prediction = model.predict(model_input, verbose=0)
        prediction_values = np.asarray(raw_prediction).reshape(-1)

        if prediction_values.size != 1:
            raise ValueError(
                "The legacy prediction path expected one temperature value, "
                f"but received {prediction_values.size} values with shape "
                f"{np.asarray(raw_prediction).shape}."
            )

        predicted_temperature = float(prediction_values[0])

        if not np.isfinite(predicted_temperature):
            raise ValueError(
                f"The model returned an invalid temperature for {target_time}: "
                f"{predicted_temperature}"
            )

        predictions.append(
            {
                "hours_ahead": hour_ahead,
                "forecast_time": target_time,
                "predicted_temperature": predicted_temperature,
            }
        )

    if not predictions:
        raise ValueError("No predictions were generated.")

    return pd.DataFrame(predictions)


def predict_weather(
    model: tf.keras.Model,
    model_type: str,
    weather: pd.DataFrame,
    current_hour: pd.Timestamp,
    requested_hours: int,
    feature_columns: list[str],
    saved_config: dict[str, Any],
) -> pd.DataFrame:
    sequence_length, expected_feature_count = get_model_dimensions(model)
    output_count = get_model_output_count(model)

    if expected_feature_count != len(feature_columns):
        raise ValueError(
            "Model feature mismatch.\n"
            f"The selected model expects {expected_feature_count} features, "
            f"but this script resolved {len(feature_columns)}.\n"
            f"Resolved feature order: {feature_columns}"
        )

    if output_count > 1:
        return predict_multi_output_model(
            model=model,
            weather=weather,
            feature_columns=feature_columns,
            current_hour=current_hour,
            requested_hours=requested_hours,
            sequence_length=sequence_length,
            output_count=output_count,
        )

    return predict_single_output_model(
        model=model,
        model_type=model_type,
        weather=weather,
        feature_columns=feature_columns,
        current_hour=current_hour,
        requested_hours=requested_hours,
        sequence_length=sequence_length,
        saved_config=saved_config,
    )


# ============================================================
# OUTPUT
# ============================================================


def print_prediction_summary(
    city: str,
    model_type: str,
    predictions: pd.DataFrame,
    timezone_name: str,
) -> None:
    display_name = str(MODEL_CONFIGS[model_type]["display_name"])

    print("\n")
    print("=" * 72)
    print(f"{display_name.upper()} WEATHER FORECAST: {city.upper()}")
    print(f"Location time zone: {timezone_name}")
    print("=" * 72)

    for row in predictions.itertuples(index=False):
        forecast_time = row.forecast_time.strftime("%A %I:%M %p")
        fahrenheit = (row.predicted_temperature * (9 / 5)) + 32

        print(
            f"{row.hours_ahead:>2} hour(s) ahead | "
            f"{forecast_time:<22} | "
            f"{row.predicted_temperature:>7.2f}{TEMPERATURE_CELSIUS} | "
            f"{fahrenheit:>7.2f}{TEMPERATURE_FAHRENHEIT}"
        )

    first_temperature = float(
        predictions.iloc[0]["predicted_temperature"]
    )
    last_temperature = float(
        predictions.iloc[-1]["predicted_temperature"]
    )
    lowest_temperature = float(
        predictions["predicted_temperature"].min()
    )
    highest_temperature = float(
        predictions["predicted_temperature"].max()
    )
    temperature_change = last_temperature - first_temperature

    if len(predictions) == 1:
        trend = "Only one forecast hour was requested."
    elif temperature_change > 1.0:
        trend = "The model expects temperatures to rise."
    elif temperature_change < -1.0:
        trend = "The model expects temperatures to fall."
    else:
        trend = (
            "The model expects temperatures to remain relatively stable."
        )

    selected_hour = int(predictions.iloc[-1]["hours_ahead"])
    selected_prediction = float(
        predictions.iloc[-1]["predicted_temperature"]
    )

    print("-" * 72)
    print(
        f"Prediction at {selected_hour} hour(s): "
        f"{selected_prediction:.2f}{TEMPERATURE_CELSIUS}"
    )
    print(
        f"Predicted range: {lowest_temperature:.2f}{TEMPERATURE_CELSIUS} "
        f"to {highest_temperature:.2f}{TEMPERATURE_CELSIUS}"
    )
    print(f"Summary: {trend}")
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("=" * 72)
    print("WEATHER MODEL — CITY FORECAST")
    print("=" * 72)

    model_type = select_model_type()
    city = select_city()
    hours = select_forecast_hours()

    latitude, longitude = CITY_COORDINATES[city]

    print("\nForecast configuration")
    print("-" * 72)
    print(f"Model:          {MODEL_CONFIGS[model_type]['display_name']}")
    print(f"City:           {city.title()}")
    print(f"Coordinates:    {latitude}, {longitude}")
    print(f"Forecast hours: {hours}")

    (
        model,
        model_path,
        model_directory,
        saved_config,
        feature_columns,
    ) = load_selected_model(model_type)

    sequence_length, feature_count = get_model_dimensions(model)
    output_count = get_model_output_count(model)

    print(f"\nModel input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Sequence length:    {sequence_length} hours")
    print(f"Feature count:      {feature_count}")
    print(f"Output count:       {output_count}")
    print(f"Feature order:      {feature_columns}")
    print("Downloading current weather features...")

    api_feature_columns = get_api_feature_columns(feature_columns)

    weather, current_hour, timezone_name = download_weather_features(
        latitude=latitude,
        longitude=longitude,
        required_history_hours=sequence_length,
        api_feature_columns=api_feature_columns,
    )

    weather = add_time_features(
        weather=weather,
        feature_columns=feature_columns,
    )

    predictions = predict_weather(
        model=model,
        model_type=model_type,
        weather=weather,
        current_hour=current_hour,
        requested_hours=hours,
        feature_columns=feature_columns,
        saved_config=saved_config,
    )

    print_prediction_summary(
        city=city,
        model_type=model_type,
        predictions=predictions,
        timezone_name=timezone_name,
    )

    print("\nPrediction details")
    print("-" * 72)
    print(f"Architecture: {MODEL_CONFIGS[model_type]['display_name']}")
    print(f"Model folder: {model_directory}")
    print(f"Model used:   {model_path.name}")
    print(f"Full path:    {model_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")
    except Exception as error:
        print("\nPREDICTION ERROR:")
        print(error)