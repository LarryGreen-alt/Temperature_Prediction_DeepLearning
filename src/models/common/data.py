from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data.preprocess import preprocess
from src.data.split_dataset import split_dataset
from src.utils.city_coordinates import CITY_COORDINATES

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_FILE = DATA_DIR / "train.csv"
DEV_FILE = DATA_DIR / "dev.csv"
TEST_FILE = DATA_DIR / "test.csv"

INPUT_HOURS = 72           # Previous 72 hours
OUTPUT_HOURS = 24          # Predict the next 24 hours, t+1h ... t+24h
STRIDE = 3                 # Window step, applied to train/dev/test alike

# Canonical order, matching src/models/lstm/model.py / models/LSTM/latest/model_config.json.
FEATURE_COLUMNS = [
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
    "day_cos"
]

TARGET_COLUMN = "temperature_2m"

TIME_COLUMN = "time"
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

    return train_df, dev_df, test_df


def split_into_contiguous_groups(dataframe, city_column=CITY_COLUMN, time_column=TIME_COLUMN):
    """Yields (city, segment_df) pairs, breaking on both city change and any
    gap larger than one hour, so no window built from a segment can span a
    city boundary or a hole in the hourly record.

    Ported from src/models/lstm/train.py's split_into_contiguous_groups."""
    working = dataframe.copy()
    working[time_column] = pd.to_datetime(working[time_column])

    for city, city_df in working.groupby(city_column, sort=True):
        city_df = city_df.sort_values(time_column).reset_index(drop=True)
        segment_ids = city_df[time_column].diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _, segment_df in city_df.groupby(segment_ids, sort=False):
            yield city, segment_df.reset_index(drop=True)


def create_multistep_sequences(
    dataframe,
    input_hours=INPUT_HOURS,
    output_hours=OUTPUT_HOURS,
    stride=STRIDE,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    with_city_ids=False,
    city_vocab=CITY_VOCAB
):
    """Returns (X, y[, city_ids]) with X (n, input_hours, F) and y (n, output_hours).

    X[i] = features[start : start+input_hours], y[i] = target[start+input_hours : start+input_hours+output_hours],
    built independently per contiguous per-city segment (see split_into_contiguous_groups)
    and stepped by `stride`."""
    minimum_rows = input_hours + output_hours
    X_parts, y_parts, city_id_parts = [], [], []

    for city, segment_df in split_into_contiguous_groups(dataframe):
        if len(segment_df) < minimum_rows:
            continue

        features = segment_df[feature_columns].to_numpy(dtype=np.float32)
        target = segment_df[target_column].to_numpy(dtype=np.float32)
        num_windows = len(segment_df) - minimum_rows + 1

        # sliding_window_view adds the window as a new trailing axis, so
        # feature_windows has shape (len(segment) - input_hours + 1, F, input_hours);
        # moveaxis puts it back in (num_windows, input_hours, F) order.
        feature_windows = sliding_window_view(features, window_shape=input_hours, axis=0)
        feature_windows = np.moveaxis(feature_windows, -1, 1)
        target_windows = sliding_window_view(target, window_shape=output_hours, axis=0)

        X_seg = feature_windows[:num_windows:stride].copy()
        y_seg = target_windows[input_hours:input_hours + num_windows:stride].copy()

        X_parts.append(X_seg)
        y_parts.append(y_seg)
        if with_city_ids:
            city_id_parts.append(np.full(len(X_seg), city_vocab[city], dtype=np.int32))

    if not X_parts:
        if with_city_ids:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    if with_city_ids:
        return X, y, np.concatenate(city_id_parts)
    return X, y
