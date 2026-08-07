"""Cross-track oracle check: score Larry's trained LSTM on the shared 16-city test split.

The LSTM is single-input 72x11 with no city id, so this is plain inference --
no architecture work needed. If the per-horizon MAE curve comes out sane (see
README/plan), the multi-horizon windowing in src/models/common/data.py is
correct; if it's flat, enormous, or non-monotonic, the windowing is wrong.

This also delivers the cross-track comparison itself: both tracks scored on
identical test windows with identical metric code (imported directly from
src.models.lstm.train), leaving city-set size as the one stated difference.

Baselines are not included yet: persistence and previous-day on the shared
windows arrive with the multi-horizon evaluation module (common/evaluation.py),
after which this file needs regenerating.
"""

import json

import numpy as np
import pandas as pd

from src.models.common.data import FEATURE_COLUMNS, PROJECT_ROOT, TEST_FILE, create_multistep_sequences
from src.models.lstm.model import load_weather_model
from src.models.lstm.train import calculate_metrics

LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "LSTM" / "latest" / "weather_lstm.keras"
OUTPUT_DIR = PROJECT_ROOT / "models" / "comparison" / "lstm_on_shared_test_split"


def main():
    print(f"Loading test split from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].astype(np.float32)
    print(f"Test rows: {len(test_df)}, cities: {test_df['city'].nunique()}")

    print("Building 72h->24h test windows in canonical feature order...")
    X_test, y_test = create_multistep_sequences(test_df)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    print(f"Loading LSTM model from {LSTM_MODEL_PATH}...")
    model = load_weather_model(LSTM_MODEL_PATH, compile=False)

    print("Running inference...")
    y_pred = model.predict(X_test, batch_size=256, verbose=1)

    model_metrics = calculate_metrics(y_test, y_pred)
    output = {
        "improved_model": model_metrics,
        "model_path": str(LSTM_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "num_cities": int(test_df["city"].nunique()),
        "num_windows": int(len(X_test)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"overall_mae={model_metrics['overall_mae']:.4f}  overall_rmse={model_metrics['overall_rmse']:.4f}  "
          f"overall_bias={model_metrics['overall_bias']:.4f}")
    print("\nmae_by_hour (h=1..24):")
    for hour, mae in enumerate(model_metrics["mae_by_hour"], start=1):
        print(f"  h={hour:2d}: {mae:.4f}")


if __name__ == "__main__":
    main()
