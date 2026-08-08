"""Fetches a live 72-hour weather window from Open-Meteo's forecast endpoint
and turns it into the 11-column feature matrix the models expect.

Training data came from the archive (ERA5 reanalysis) endpoint, which lags
real time by days, so it can't serve "the last 72 hours up to now". The
forecast endpoint's past_hours parameter can, at the cost of being a
slightly different data source (recent observations/short-range analysis,
not reanalysis) -- see demo/PLAN.md for the full rationale.
"""

import time

import numpy as np
import pandas as pd
import requests

from src.models.common.data import FEATURE_COLUMNS, INPUT_HOURS
from src.utils.city_coordinates import CITY_COORDINATES

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = (
    "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,"
    "cloud_cover,precipitation,is_day"
)

CACHE_TTL_SECONDS = 30 * 60
_cache = {}


def _fetch_hourly(city_key):
    latitude, longitude = CITY_COORDINATES[city_key]
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "past_hours": INPUT_HOURS,
            "forecast_days": 1,
            "timezone": "UTC",
            "hourly": HOURLY_VARIABLES,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["hourly"]


def _engineer_features(hourly):
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna()

    # Only keep hours that have actually happened -- past_hours + forecast_days
    # returns some hours still ahead of "now", which we don't want in the
    # input window.
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    df = df[df["time"] <= now]
    df = df.sort_values("time").tail(INPUT_HOURS).reset_index(drop=True)

    if len(df) < INPUT_HOURS:
        raise ValueError(
            f"Open-Meteo returned only {len(df)} usable hours, need {INPUT_HOURS}."
        )

    hour = df["time"].dt.hour
    day_of_year = df["time"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    return df


def get_live_window(city_key):
    """Returns (features, last_observed) for one city.

    features: (INPUT_HOURS, len(FEATURE_COLUMNS)) float32 array, ready to
    feed a model. last_observed: list of {"time", "temperature_c"} dicts for
    the same window, for the frontend to plot as the "observed" line.
    """
    cached = _cache.get(city_key)
    if cached is not None and time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS:
        return cached["features"], cached["last_observed"]

    hourly = _fetch_hourly(city_key)
    df = _engineer_features(hourly)

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    last_observed = [
        {"time": row.time.isoformat(), "temperature_c": float(row.temperature_2m)}
        for row in df.itertuples()
    ]

    _cache[city_key] = {
        "features": features,
        "last_observed": last_observed,
        "fetched_at": time.time(),
    }
    return features, last_observed
