from __future__ import annotations

import os

# Suppress TensorFlow INFO messages before TensorFlow/model.py is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from .model import load_weather_model
except ImportError:
    from model import load_weather_model

try:
    from src.utils.city_coordinates import (
        CITY_COORDINATES,
        get_coordinates,
    )
except ImportError:
    CITY_COORDINATES = {}

    def get_coordinates(city_name: str):
        raise KeyError(
            "City-coordinate helpers could not be imported. "
            "Provide --latitude and --longitude."
        )


TIME_COLUMN_CANDIDATES = ["time", "datetime", "timestamp", "date"]
GROUP_COLUMN_CANDIDATES = ["city", "location", "location_name"]


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LIVE_API_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
]


def detect_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    return next(
        (name for name in candidates if name in dataframe.columns),
        None,
    )


def celsius_to_fahrenheit(
    values: np.ndarray | pd.Series,
) -> np.ndarray | pd.Series:
    """Convert Celsius temperatures to Fahrenheit."""
    return values * 9.0 / 5.0 + 32.0


def discover_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd(), script_dir, *script_dir.parents]

    for candidate in candidates:
        if (candidate / "models" / "LSTM" / "latest").exists():
            return candidate

    return Path.cwd()


def choose_live_city(
    requested_city: str | None,
) -> tuple[str, float, float]:
    """Choose a configured city and return its latitude and longitude."""
    if requested_city:
        normalized = requested_city.strip().lower()
        if CITY_COORDINATES:
            lookup = {
                str(name).lower(): str(name)
                for name in CITY_COORDINATES
            }
            matched = lookup.get(normalized)
            if matched is None:
                raise ValueError(
                    f"Unknown city '{requested_city}'. Available cities: "
                    f"{', '.join(sorted(map(str, CITY_COORDINATES)))}"
                )
            latitude, longitude = get_coordinates(matched)
            return matched, float(latitude), float(longitude)

        raise ValueError(
            "--city requires src.utils.city_coordinates, or provide "
            "--latitude and --longitude."
        )

    if not CITY_COORDINATES:
        raise ValueError(
            "No city list was available. Provide --latitude and --longitude."
        )

    cities = sorted(map(str, CITY_COORDINATES))
    print("\nAvailable live forecast cities\n")
    for index, city in enumerate(cities, start=1):
        print(f" {index}. {city.title()}")

    while True:
        entered = input(
            f"\nChoose a city (1-{len(cities)}) or type its name: "
        ).strip()

        try:
            index = int(entered)
        except ValueError:
            index = -1

        if 1 <= index <= len(cities):
            city = cities[index - 1]
            break

        lookup = {city.lower(): city for city in cities}
        city = lookup.get(entered.lower())
        if city is not None:
            break

        print("Choose a number or city from the available list.")

    latitude, longitude = get_coordinates(city)
    return city, float(latitude), float(longitude)


def fetch_live_weather_history(
    *,
    city_name: str,
    latitude: float,
    longitude: float,
    input_hours: int,
) -> tuple[pd.DataFrame, str, pd.Timestamp]:
    """
    Fetch recent hourly inputs through the current local hour.

    The API also returns future hours for the current day. They are deliberately
    removed so the LSTM sees only information available up to 'now'. The model
    forecast therefore begins at the location's next complete clock hour.
    """
    past_days = max(4, int(np.ceil(input_hours / 24.0)) + 1)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(LIVE_API_COLUMNS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    request_url = f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={"User-Agent": "TemperaturePredictionDeepLearning/1.0"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Could not download recent weather inputs for "
            f"{city_name}: {exc}"
        ) from exc

    if "error" in payload:
        raise RuntimeError(
            f"Weather service error: {payload.get('reason', payload['error'])}"
        )

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(
            "The weather service response did not contain hourly data."
        )

    timezone_name = payload.get("timezone", "UTC")
    try:
        location_timezone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unsupported timezone returned by the weather service: "
            f"{timezone_name}"
        ) from exc

    dataframe = pd.DataFrame(hourly)
    dataframe["city"] = city_name
    dataframe["time"] = pd.to_datetime(
        dataframe["time"],
        errors="coerce",
    )

    if dataframe["time"].isna().any():
        raise ValueError(
            "The live weather response contained invalid hourly timestamps."
        )

    now_local = pd.Timestamp.now(tz=location_timezone)
    current_hour_local = now_local.floor("h")

    # Open-Meteo returns local wall-clock strings when timezone=auto.
    local_naive_cutoff = current_hour_local.tz_localize(None)
    dataframe = dataframe[
        dataframe["time"] <= local_naive_cutoff
    ].copy()

    if len(dataframe) < input_hours:
        raise ValueError(
            f"Only {len(dataframe)} recent hourly rows were available; "
            f"the model requires {input_hours}."
        )

    dataframe = add_derived_time_features(dataframe)
    dataframe = dataframe.sort_values("time").reset_index(drop=True)

    latest_api_time = pd.Timestamp(dataframe["time"].iloc[-1])
    if latest_api_time < local_naive_cutoff:
        print(
            "NOTE: The weather service's newest available hourly input is "
            f"{latest_api_time}, which is earlier than the current local hour "
            f"{local_naive_cutoff}."
        )

    print(
        f"\nDownloaded recent inputs for {city_name.title()} "
        f"({timezone_name})."
    )
    print(f"Current local time : {now_local.strftime('%Y-%m-%d %I:%M %p %Z')}")
    print(f"Latest input hour  : {latest_api_time.strftime('%Y-%m-%d %I:%M %p')}")

    return dataframe, timezone_name, current_hour_local


def discover_input_csvs(project_root: Path) -> list[Path]:
    """
    Find likely inference files so input_csv is optional and interactive.
    Processed files are preferred because they should contain model features.
    """
    ordered_patterns = [
        project_root / "data" / "processed" / "*.csv",
        project_root / "data" / "raw" / "*.csv",
        project_root / "data" / "splits" / "test.csv",
    ]

    discovered: list[Path] = []
    for pattern in ordered_patterns:
        for path in sorted(pattern.parent.glob(pattern.name)):
            resolved = path.resolve()
            if resolved not in discovered:
                discovered.append(resolved)

    return discovered


def choose_input_csv(
    supplied_path: Path | None,
    project_root: Path,
) -> Path:
    if supplied_path is not None:
        return supplied_path.resolve()

    candidates = discover_input_csvs(project_root)
    if not candidates:
        entered = input(
            "Enter the CSV path containing the latest weather observations: "
        ).strip()
        if not entered:
            raise ValueError("An input CSV is required.")
        return Path(entered).expanduser().resolve()

    if len(candidates) == 1:
        print(f"Using input CSV: {candidates[0]}")
        return candidates[0]

    print("\nAvailable input CSV files\n")
    for index, path in enumerate(candidates, start=1):
        try:
            shown = path.relative_to(project_root)
        except ValueError:
            shown = path
        print(f" {index}. {shown}")

    while True:
        entered = input(
            f"\nChoose a file (1-{len(candidates)}): "
        ).strip()
        try:
            choice = int(entered)
        except ValueError:
            print("Enter a number from the list.")
            continue

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]

        print("Choice is outside the available range.")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_derived_time_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create cyclical time features when a raw weather CSV is selected.

    Training uses hour_sin/hour_cos and day_sin/day_cos. Raw downloads often
    contain only a timestamp, so inference must reproduce those features before
    validating the model input.
    """
    required_time_features = {
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    }
    missing = required_time_features.difference(dataframe.columns)

    if not missing:
        return dataframe.copy()

    time_column = detect_column(
        dataframe,
        TIME_COLUMN_CANDIDATES,
    )
    if time_column is None:
        raise ValueError(
            "The selected CSV is missing cyclical time features "
            f"{sorted(missing)} and has no timestamp column from which they "
            "can be calculated. Choose a processed CSV or add a time/datetime/"
            "timestamp/date column."
        )

    working = dataframe.copy()
    timestamps = pd.to_datetime(
        working[time_column],
        errors="coerce",
    )

    bad_count = int(timestamps.isna().sum())
    if bad_count:
        raise ValueError(
            f"Cannot create time features because column '{time_column}' "
            f"contains {bad_count} invalid timestamps."
        )

    # Match the feature formulas already used by the project during training.
    hour = timestamps.dt.hour.astype(float)
    day_of_year = timestamps.dt.dayofyear.astype(float)

    hour_angle = 2.0 * np.pi * hour / 24.0
    day_angle = 2.0 * np.pi * day_of_year / 365.25

    if "hour_sin" not in working.columns:
        working["hour_sin"] = np.sin(hour_angle)
    if "hour_cos" not in working.columns:
        working["hour_cos"] = np.cos(hour_angle)
    if "day_sin" not in working.columns:
        working["day_sin"] = np.sin(day_angle)
    if "day_cos" not in working.columns:
        working["day_cos"] = np.cos(day_angle)

    print(
        "Generated missing time features from "
        f"'{time_column}': {', '.join(sorted(missing))}"
    )
    return working


def validate_input(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "Input CSV is still missing required model columns after automatic "
            f"feature preparation: {missing}\n"
            "Choose data/processed/features.csv, or make sure the raw file "
            "contains the weather measurements used during training."
        )

    dataframe = dataframe.copy()

    for column in feature_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid = dataframe[feature_columns].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise ValueError(
            "Input CSV contains missing/non-numeric feature values:\n"
            f"{invalid.to_string()}"
        )

    dataframe[feature_columns] = dataframe[feature_columns].astype(
        np.float32
    )
    return dataframe


def select_location(
    dataframe: pd.DataFrame,
    requested_city: str | None,
) -> tuple[pd.DataFrame, str | None]:
    group_column = detect_column(
        dataframe,
        GROUP_COLUMN_CANDIDATES,
    )
    if group_column is None:
        return dataframe.copy(), None

    available = [
        str(value)
        for value in dataframe[group_column].dropna().unique()
    ]
    if not available:
        raise ValueError(
            f"Column '{group_column}' contains no locations."
        )

    city = requested_city
    if city is None:
        if len(available) == 1:
            city = available[0]
        else:
            print("\nAvailable locations\n")
            for value in available:
                print(f" - {value}")
            city = input("\nEnter city/location: ").strip()

    lookup = {
        value.casefold(): value
        for value in available
    }
    matched = lookup.get(city.casefold()) if city else None

    if matched is None:
        raise ValueError(
            f"Location '{city}' was not found. "
            f"Available values: {available}"
        )

    selected = dataframe[
        dataframe[group_column].astype(str) == matched
    ].copy()
    return selected, matched


def latest_contiguous_window(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    input_hours: int,
) -> tuple[np.ndarray, pd.Timestamp | None]:
    time_column = detect_column(
        dataframe,
        TIME_COLUMN_CANDIDATES,
    )
    working = dataframe.copy()

    if time_column is None:
        if len(working) < input_hours:
            raise ValueError(
                f"Input requires at least {input_hours} rows; "
                f"found {len(working)}."
            )

        window = working.iloc[-input_hours:]
        return (
            window[feature_columns].to_numpy(dtype=np.float32),
            None,
        )

    working[time_column] = pd.to_datetime(
        working[time_column],
        errors="coerce",
    )
    bad_count = int(working[time_column].isna().sum())
    if bad_count:
        raise ValueError(
            f"Input has {bad_count} invalid timestamps "
            f"in '{time_column}'."
        )

    working = working.sort_values(
        time_column
    ).reset_index(drop=True)
    new_segment = (
        working[time_column]
        .diff()
        .ne(pd.Timedelta(hours=1))
    )
    working["_segment"] = new_segment.cumsum()

    usable_segments = [
        segment.drop(
            columns="_segment"
        ).reset_index(drop=True)
        for _, segment in working.groupby(
            "_segment",
            sort=False,
        )
        if len(segment) >= input_hours
    ]

    if not usable_segments:
        raise ValueError(
            f"No uninterrupted hourly segment has {input_hours} rows."
        )

    latest_segment = max(
        usable_segments,
        key=lambda frame: frame[time_column].iloc[-1],
    )
    window = latest_segment.iloc[-input_hours:]
    last_timestamp = pd.Timestamp(
        window[time_column].iloc[-1]
    )

    return (
        window[feature_columns].to_numpy(dtype=np.float32),
        last_timestamp,
    )


def parse_args() -> argparse.Namespace:
    project_root = discover_project_root()
    latest_dir = project_root / "models" / "LSTM" / "latest"

    parser = argparse.ArgumentParser(
        description=(
            "Generate a 1-24 hour forecast. By default, recent hourly inputs "
            "are downloaded through the selected city's current local hour, "
            "so hour 1 begins at the next local clock hour. Pass an input CSV "
            "only when you intentionally want file-based inference."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional historical/model-ready CSV. Omit it for a forecast "
            "starting from today's current local hour."
        ),
    )
    parser.add_argument(
        "--file-mode",
        action="store_true",
        help=(
            "Choose an existing CSV interactively instead of downloading "
            "recent inputs. File-mode forecasts begin after that file's last "
            "timestamp."
        ),
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=None,
        help="Latitude for live mode when not using a configured city.",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="Longitude for live mode when not using a configured city.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=latest_dir / "weather_lstm.keras",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=latest_dir / "model_config.json",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Forecast length from 1 through 24 hours.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default=None,
        help=(
            "Optional IANA timezone for displaying forecast times, such as "
            "Europe/Rome or America/Chicago. When omitted, timestamps are "
            "treated as the location-local times already stored in the CSV."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = discover_project_root()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Trained model not found: {args.model}"
        )

    config = load_config(args.config)
    input_hours = int(config["input_hours"])
    output_hours = int(config["output_hours"])
    feature_columns = list(config["feature_columns"])

    use_file_mode = args.file_mode or args.input_csv is not None
    input_csv = None
    live_timezone_name = None
    current_local_hour = None
    live_city_name = None

    if use_file_mode:
        input_csv = choose_input_csv(
            args.input_csv,
            project_root,
        )
        if not input_csv.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {input_csv}"
            )
    else:
        if (args.latitude is None) != (args.longitude is None):
            raise ValueError(
                "Provide both --latitude and --longitude, or neither."
            )

        if args.latitude is not None and args.longitude is not None:
            live_city_name = args.city or "selected location"
            latitude = float(args.latitude)
            longitude = float(args.longitude)
        else:
            live_city_name, latitude, longitude = choose_live_city(
                args.city
            )

        dataframe, live_timezone_name, current_local_hour = (
            fetch_live_weather_history(
                city_name=live_city_name,
                latitude=latitude,
                longitude=longitude,
                input_hours=input_hours,
            )
        )

    requested_hours = args.hours
    if requested_hours is None:
        entered = input(
            f"Forecast how many hours (1-{output_hours})? "
        ).strip()
        requested_hours = (
            int(entered)
            if entered
            else output_hours
        )

    if not 1 <= requested_hours <= output_hours:
        raise ValueError(
            f"Forecast hours must be between 1 and {output_hours}."
        )

    if use_file_mode:
        dataframe = pd.read_csv(input_csv)
        dataframe = add_derived_time_features(dataframe)
        dataframe, selected_city = select_location(
            dataframe,
            args.city,
        )
    else:
        selected_city = live_city_name

    dataframe = validate_input(
        dataframe,
        feature_columns,
    )
    input_window, last_timestamp = latest_contiguous_window(
        dataframe,
        feature_columns,
        input_hours,
    )

    model = load_weather_model(
        args.model,
        compile=False,
    )
    prediction = np.asarray(
        model.predict(
            input_window[np.newaxis, ...],
            verbose=0,
        )[0],
        dtype=float,
    )

    predicted_c = prediction[:requested_hours]
    predicted_f = celsius_to_fahrenheit(predicted_c)

    records: dict[str, object] = {
        "forecast_hour": np.arange(
            1,
            requested_hours + 1,
        ),
        "predicted_temperature_c": predicted_c,
        "predicted_temperature_f": predicted_f,
    }

    if last_timestamp is not None:
        # Begin at the next complete clock hour for the selected location.
        # Example: a final observation at 16:00 or 16:37 starts at 17:00.
        forecast_start = (
            pd.Timestamp(last_timestamp).floor("h")
            + pd.Timedelta(hours=1)
        )

        forecast_times = pd.date_range(
            start=forecast_start,
            periods=requested_hours,
            freq="h",
        )

        display_timezone = args.timezone or live_timezone_name
        if display_timezone:
            try:
                if forecast_times.tz is None:
                    forecast_times = forecast_times.tz_localize(
                        display_timezone,
                        ambiguous="infer",
                        nonexistent="shift_forward",
                    )
                else:
                    forecast_times = forecast_times.tz_convert(
                        display_timezone
                    )
            except Exception as exc:
                raise ValueError(
                    f"Could not apply timezone '{display_timezone}': {exc}"
                ) from exc

        records["forecast_time_local"] = forecast_times

    forecast_df = pd.DataFrame(records)

    if "forecast_time_local" in forecast_df.columns:
        forecast_df = forecast_df[
            [
                "forecast_hour",
                "forecast_time_local",
                "predicted_temperature_c",
                "predicted_temperature_f",
            ]
        ]

    location_text = (
        selected_city
        if selected_city is not None
        else input_csv.stem
    )

    print(f"\nTemperature forecast for {location_text}")

    if last_timestamp is not None:
        source_time = pd.Timestamp(last_timestamp)
        next_hour = source_time.floor("h") + pd.Timedelta(hours=1)
        time_label = (
            args.timezone
            or live_timezone_name
            or "location-local time from CSV"
        )
        print(
            f"Last observation : {source_time} "
            f"({time_label})"
        )
        print(
            f"Forecast starts  : {next_hour} "
            f"({time_label})"
        )

        if use_file_mode:
            # This warning applies only to explicitly requested file inference.
            now_naive = pd.Timestamp.now().tz_localize(None)
            source_naive = (
                source_time.tz_localize(None)
                if source_time.tzinfo
                else source_time
            )
            data_age_hours = (
                now_naive - source_naive
            ).total_seconds() / 3600.0

            if data_age_hours > 48:
                print(
                    "WARNING: File mode is using stale historical data. "
                    "Run predict.py without an input CSV for a forecast "
                    "starting from today's next local hour."
                )
        elif current_local_hour is not None:
            expected_start = current_local_hour + pd.Timedelta(hours=1)
            actual_start = (
                pd.Timestamp(last_timestamp).floor("h")
                + pd.Timedelta(hours=1)
            )
            if actual_start.tzinfo is None:
                actual_start = actual_start.tz_localize(
                    live_timezone_name
                )
            if actual_start != expected_start:
                print(
                    "NOTE: The live service's latest available input hour "
                    "lags the current local hour, so the first forecast time "
                    f"is {actual_start}."
                )

    print("-" * 96)
    print(
        forecast_df.to_string(
            index=False,
            formatters={
                "forecast_time_local": (
                    lambda value: pd.Timestamp(value).strftime(
                        "%Y-%m-%d %I:%M %p %Z"
                    ).strip()
                ),
                "predicted_temperature_c": (
                    lambda value: f"{value:.2f} °C"
                ),
                "predicted_temperature_f": (
                    lambda value: f"{value:.2f} °F"
                ),
            },
        )
    )

    output_path = args.output_csv
    if output_path is None:
        output_dir = project_root / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_location = str(location_text).strip().lower().replace(
            " ",
            "_",
        )
        output_path = (
            output_dir
            / f"{safe_location}_temperature_forecast.csv"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    forecast_df.to_csv(
        output_path,
        index=False,
    )
    print(f"\nForecast saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()