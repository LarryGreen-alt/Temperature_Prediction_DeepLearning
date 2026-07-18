from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def load_splits(train_file=TRAIN_FILE, dev_file=DEV_FILE, test_file=TEST_FILE):
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
