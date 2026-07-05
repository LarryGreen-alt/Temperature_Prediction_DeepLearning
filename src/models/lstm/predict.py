import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

WINDOW_SIZE = 24

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "weather_lstm.keras"
TEST_DATA = DATA_DIR / "splits" / "test.csv"

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


MODEL_PATH = "models/weather_lstm.keras"

TEST_DATA = "data/splits/test.csv"


def create_sequences(df):

    X = []

    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    for i in range(len(df) - WINDOW_SIZE + 1):

        X.append(
            features[
                i:i + WINDOW_SIZE
            ]
        )

    return np.array(X)


def main():

    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading data...")

    df = pd.read_csv(TEST_DATA)

    X = create_sequences(df)

    predictions = model.predict(
        X,
        verbose=0
    )

    print()

    print("First 10 Predictions")

    print("----------------------------")

    for i in range(10):

        print(
            f"Prediction {i+1}: "
            f"{predictions[i][0]:.2f} °C"
        )


if __name__ == "__main__":

    main()