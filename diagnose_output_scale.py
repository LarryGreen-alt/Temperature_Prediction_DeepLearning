"""
Ten-minute diagnostic: is the Transformer's error a generalisation problem,
or a train/inference mismatch in the output scale?

Run from the repository root:

    python -m diagnose_output_scale          # if you drop this file in the repo root
    # or just: python diagnose_output_scale.py

What it does. Keras reports training loss with dropout ACTIVE, and validation
loss with dropout OFF. Our training loss (MSE 1.61) and validation loss (MSE
6.66) are therefore measured in two different regimes, so the gap between them
does NOT cleanly mean "high variance". This script re-scores the saved model on
the SAME training windows with dropout OFF, which puts train and dev on equal
footing and settles the question.

How to read the result:

  * If train MSE (inference mode) stays near 1.6 and the bias stays near 0,
    the gap to dev is genuine high variance -> the fix is regularisation,
    more data, or fewer parameters.

  * If train MSE (inference mode) jumps toward 5-6 with a bias near -1.5,
    the model is mis-scaled at inference and the gap has nothing to do with
    generalisation -> the fix is in the head (dropout placement, output
    parameterisation, longer training), not regularisation.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.common import data as D

EXPERIMENT = Path("models/Transformer/exp03_city_and_temperature_features/"
                  "experiments/2026-08-01_14-56-03")
MODEL_FILE = EXPERIMENT / "weather_transformer.keras"

# exp03's inputs: baseline features plus the target's own history, city-aware.
FEATURES = D.FEATURE_COLUMNS + [D.TARGET_COLUMN]
SAMPLE = 200_000          # windows to score per split; None for all


def score(model, X, city, y, label):
    inputs = [X, city]
    pred = model.predict(inputs, batch_size=1024, verbose=0).flatten()
    err = pred - y
    slope, intercept = np.polyfit(y, pred, 1)
    print(f"{label:<26s} n={len(y):>7,}  "
          f"MSE {np.mean(err ** 2):7.3f}  MAE {np.mean(np.abs(err)):6.3f}  "
          f"RMSE {np.sqrt(np.mean(err ** 2)):6.3f}  "
          f"bias {err.mean():+6.3f}  slope {slope:5.3f}  intercept {intercept:+6.3f}")
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
            "bias": float(err.mean()), "slope": float(slope),
            "intercept": float(intercept)}


def main():
    model = tf.keras.models.load_model(MODEL_FILE)
    train_df, dev_df, test_df = D.load_splits()

    rng = np.random.default_rng(0)
    results = {}

    print()
    print("Scored with dropout OFF (Keras inference mode) on every split:")
    print("-" * 108)

    for name, df in [("TRAIN (inference mode)", train_df),
                     ("DEV", dev_df),
                     ("TEST", test_df)]:
        X, y, city = D.create_city_aware_sequences(
            df, D.WINDOW_SIZE, D.FORECAST_HORIZON, FEATURES, D.TARGET_COLUMN
        )
        if SAMPLE is not None and len(y) > SAMPLE:
            idx = rng.choice(len(y), SAMPLE, replace=False)
            X, y, city = X[idx], y[idx], city[idx]
        results[name] = score(model, X, city, y, name)

    print("-" * 108)
    print()

    train, dev = results["TRAIN (inference mode)"], results["DEV"]
    gap = dev["mse"] - train["mse"]
    bias_share = train["bias"] ** 2 / train["mse"] * 100

    print(f"Train MSE with dropout off : {train['mse']:.3f}   "
          f"(Keras reported {1.610:.3f} with dropout ON during training)")
    print(f"Train -> dev MSE gap       : {gap:+.3f}")
    print(f"Bias share of train MSE    : {bias_share:.1f}%")
    print()

    if abs(train["bias"]) > 0.8 or train["slope"] < 0.93:
        print("VERDICT: mis-scaled output. The distortion is present on the very data")
        print("the model was trained on, so it is NOT a generalisation failure. Fix the")
        print("head, not the regularisation. Start by moving or removing the Dropout")
        print("between the two ReLU Dense layers in architecture.py, then retrain and")
        print("re-run this script.")
    else:
        print("VERDICT: genuine variance. The model fits training data cleanly and")
        print("degrades on unseen data. Fix with regularisation, more data, or a")
        print("smaller model.")

    with open("diagnose_output_scale.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote diagnose_output_scale.json")


if __name__ == "__main__":
    main()
