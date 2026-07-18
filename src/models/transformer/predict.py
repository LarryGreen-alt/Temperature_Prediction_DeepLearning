import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from src.models.transformer.data import FEATURE_COLUMNS

WINDOW_SIZE = 24

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "Transformer" / "checkpoints" / "best.keras"
TEST_DATA = DATA_DIR / "splits" / "test.csv"


def create_sequences(df):

    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    X = [features[i:i + WINDOW_SIZE] for i in range(len(df) - WINDOW_SIZE + 1)]

    return np.array(X)


def main():

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading data...")
    df = pd.read_csv(TEST_DATA)
    X = create_sequences(df)

    predictions = model.predict(X, verbose=0)

    print()
    print("First 10 Predictions")
    print("----------------------------")

    for i in range(10):
        print(f"Prediction {i+1}: {predictions[i][0]:.2f} °C")


if __name__ == "__main__":
    main()
