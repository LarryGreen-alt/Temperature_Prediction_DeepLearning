from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import preprocess
from src.data.split_dataset import split_dataset
from src.utils.city_coordinates import CITY_COORDINATES

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_FILE = DATA_DIR / "train.csv"
DEV_FILE = DATA_DIR / "dev.csv"
TEST_FILE = DATA_DIR / "test.csv"

WINDOW_SIZE = 24          # Previous 24 hours
FORECAST_HORIZON = 3      # Predict 3 hours ahead

FEATURE_COLUMNS = [
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "is_day",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

TARGET_COLUMN = "temperature_2m"

CITY_COLUMN = "city"
CITY_VOCAB = {city: idx for idx, city in enumerate(sorted(CITY_COORDINATES))}
NUM_CITIES = len(CITY_VOCAB)


def _ensure_splits_exist(train_file, dev_file, test_file):
    if train_file.exists() and dev_file.exists() and test_file.exists():
        return

    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.csv")):
        raise FileNotFoundError(
            f"No split files under {DATA_DIR} and no raw CSVs under {RAW_DIR}. "
            "If this repo tracks data/raw/ via Git LFS, run `git lfs pull` first. "
            "Otherwise, fetch each city's raw weather history (see "
            "src/data/collect_historical.py and src/utils/city_coordinates.py), "
            "then this function can regenerate the splits automatically."
        )

    print(f"{DATA_DIR} not found -- regenerating from {RAW_DIR} (preprocess + split, ~20s)...")
    preprocess()
    split_dataset()


def load_splits(train_file=TRAIN_FILE, dev_file=DEV_FILE, test_file=TEST_FILE):
    _ensure_splits_exist(train_file, dev_file, test_file)

    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)

    for df in [train_df, dev_df, test_df]:
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(np.float32)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.float32)

    return train_df, dev_df, test_df


def create_sequences(
    dataframe,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN
):
    X = []
    y = []

    features = dataframe[feature_columns].values
    target = dataframe[target_column].values

    for i in range(len(dataframe) - window_size - forecast_horizon + 1):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size + forecast_horizon - 1])

    return np.array(X), np.array(y)


def create_city_aware_sequences(
    dataframe,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    city_column=CITY_COLUMN,
    city_vocab=CITY_VOCAB
):
    """Like create_sequences(), but builds each city's windows independently
    so a window never spans two cities, and returns a parallel array of
    integer city ids (one per window) alongside X/y. Reuses create_sequences()
    per city group, so the weather/target windows themselves are identical to
    what create_sequences() would produce for that city's rows alone.

    Lives here (not in a model-specific module) so any architecture's
    experiments can build identical city ids and windows for a fair
    comparison."""
    X_parts, y_parts, city_id_parts = [], [], []

    for city, group in dataframe.groupby(city_column, sort=True):
        X_city, y_city = create_sequences(
            group, window_size, forecast_horizon, feature_columns, target_column
        )
        if len(X_city) == 0:
            continue
        X_parts.append(X_city)
        y_parts.append(y_city)
        city_id_parts.append(np.full(len(X_city), city_vocab[city], dtype=np.int32))

    return np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(city_id_parts)
