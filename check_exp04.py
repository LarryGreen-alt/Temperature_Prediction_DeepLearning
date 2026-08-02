"""Verdict on H1: did zeroing the head dropout remove the output-scale distortion?

Reads predictions.csv from the newest exp03 (control) and exp04 (experiment)
experiment directories and reports the calibration of each. Run from the
repository root:

    python check_exp04.py

H1 is accepted only if BOTH primary criteria hold for exp04: abs(bias) < 0.5 °C
and slope > 0.95. Test MAE below 1.5 is a secondary reference point (hand-
recalibrating exp03 gives 0.92), and correlation staying above 0.99 is a
guardrail -- if accuracy improves while correlation drops, the model traded
structure for calibration.
"""

import numpy as np
import pandas as pd

from src.models.common.compare import latest_experiment_with_metrics

RUNS = [("CONTROL    exp03", "Transformer/exp03_city_and_temperature_features"),
        ("EXPERIMENT exp04", "Transformer/exp04_head_dropout")]


def summarise(label, model_name):
    experiment_dir = latest_experiment_with_metrics(model_name)
    if experiment_dir is None:
        print(f"{label}: no experiment with metrics.json found -- has it been trained?")
        return None

    df = pd.read_csv(experiment_dir / "predictions.csv")
    actual, predicted = df["Actual"].values, df["Predicted"].values
    error = predicted - actual
    slope, intercept = np.polyfit(actual, predicted, 1)
    correlation = np.corrcoef(actual, predicted)[0, 1]

    print(f"{label}  ({experiment_dir.name})")
    print(f"    n {len(df):>7,}   MAE {np.mean(np.abs(error)):6.3f}   "
          f"RMSE {np.sqrt(np.mean(error ** 2)):6.3f}   bias {error.mean():+6.3f}")
    print(f"    slope {slope:6.3f}   intercept {intercept:+6.3f}   "
          f"correlation {correlation:7.5f}")

    return {"mae": np.mean(np.abs(error)), "bias": error.mean(),
            "slope": slope, "correlation": correlation}


def main():
    print()
    results = {label: summarise(label, name) for label, name in RUNS}
    print()

    experiment = results["EXPERIMENT exp04"]
    if experiment is None:
        return

    checks = [("bias   abs(bias) < 0.5 ", abs(experiment["bias"]) < 0.5, True),
              ("slope  > 0.95         ", experiment["slope"] > 0.95, True),
              ("MAE    < 1.5          ", experiment["mae"] < 1.5, False),
              ("corr   > 0.99         ", experiment["correlation"] > 0.99, False)]

    for name, passed, primary in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              f"{'(primary)' if primary else '(secondary/guardrail)'}")

    accepted = all(passed for _, passed, primary in checks if primary)
    print(f"\nH1 {'ACCEPTED' if accepted else 'REJECTED'}: "
          f"the head dropout {'was' if accepted else 'was not'} the cause of the "
          "output-scale distortion.\n")


if __name__ == "__main__":
    main()
