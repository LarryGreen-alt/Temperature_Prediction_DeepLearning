"""Loads both models once at import time and runs predictions.

TensorFlow model loading is slow enough that doing it per request would make
the demo feel broken, so this module is imported once by main.py at startup
and the loaded models are reused for every /api/predict call.
"""

import numpy as np
import tensorflow as tf

from src.models.common.data import CITY_VOCAB, PROJECT_ROOT
from src.models.lstm.model import load_weather_model


# models/LSTM/checkpoints/best.keras is a stale 24h/10-feature checkpoint
# left over from an early baseline run (git commit 6fbd411) -- it predates
# the switch to the 72h/11-feature architecture and will not load against
# this pipeline's input window. models/LSTM/latest/weather_lstm.keras is the
# current best 72h/11-feature model (see its model_config.json).
LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "LSTM" / "latest" / "weather_lstm.keras"
TRANSFORMER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Transformer"
    / "exp04_baseline_blend"
    / "checkpoints"
    / "best.keras"
)

print(f"Loading LSTM model: {LSTM_MODEL_PATH}")
_lstm_model = load_weather_model(LSTM_MODEL_PATH, compile=False)

print(f"Loading Transformer model: {TRANSFORMER_MODEL_PATH}")
_transformer_model = tf.keras.models.load_model(TRANSFORMER_MODEL_PATH, compile=False)

print("Both models loaded.")


def predict(model_name, city_key, features):
    """features: (INPUT_HOURS, len(FEATURE_COLUMNS)) array for one city.

    Returns a length-OUTPUT_HOURS list of predicted temperatures (Celsius).
    """
    X = features[np.newaxis, ...]

    if model_name == "lstm":
        raw = _lstm_model.predict(X, verbose=0)
    elif model_name == "transformer":
        city_id = np.array([CITY_VOCAB[city_key]], dtype=np.int32)
        raw = _transformer_model.predict([X, city_id], verbose=0)
    else:
        raise ValueError(f"Unknown model '{model_name}', expected 'lstm' or 'transformer'.")

    return np.asarray(raw, dtype=float)[0].tolist()
