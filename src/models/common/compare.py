"""Builds a side-by-side LSTM vs Transformer metrics comparison for the report.

Loads each model's most recent experiment (the one with the highest timestamp
that contains a metrics.json), writes a comparison table + bar chart to
models/comparison/.
"""

import json

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_NAMES = ["LSTM", "Transformer"]

OUTPUT_DIR = MODEL_DIR / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def latest_experiment_with_metrics(model_name):
    experiments_dir = MODEL_DIR / model_name / "experiments"

    if not experiments_dir.exists():
        return None

    candidates = sorted((d for d in experiments_dir.iterdir() if d.is_dir()), reverse=True)

    for experiment_dir in candidates:
        if (experiment_dir / "metrics.json").exists():
            return experiment_dir

    return None


def find_cached_experiment(model_name, config):
    """Scans models/<model_name>/experiments/*/ newest-first. Returns the
    first directory whose config.json exists and exactly matches `config`.
    Returns None if no prior run used this exact hyperparameter combination
    (this includes experiments predating config.json, which won't match
    anything since the file is simply missing)."""

    experiments_dir = MODEL_DIR / model_name / "experiments"

    if not experiments_dir.exists():
        return None

    candidates = sorted((d for d in experiments_dir.iterdir() if d.is_dir()), reverse=True)

    for experiment_dir in candidates:
        config_file = experiment_dir / "config.json"

        if not config_file.exists():
            continue

        with open(config_file) as f:
            if json.load(f) == config:
                return experiment_dir

    return None


def load_row(model_name):
    experiment_dir = latest_experiment_with_metrics(model_name)

    if experiment_dir is None:
        print(f"No experiment with metrics.json found for {model_name}, skipping.")
        return None

    with open(experiment_dir / "metrics.json") as f:
        metrics = json.load(f)

    epochs_trained = None

    history_file = experiment_dir / "training_history.csv"

    if history_file.exists():
        epochs_trained = len(pd.read_csv(history_file))

    return {
        "model": model_name,
        "experiment": experiment_dir.name,
        "loss": metrics.get("loss"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "epochs_trained": epochs_trained
    }


def compare(model_names=MODEL_NAMES):
    rows = [
        row
        for row in (load_row(model_name) for model_name in model_names)
        if row is not None
    ]

    if not rows:
        raise FileNotFoundError(
            "No experiments with metrics.json found for any model. "
            "Train the LSTM and Transformer models first."
        )

    comparison_df = pd.DataFrame(rows)

    comparison_file = OUTPUT_DIR / "comparison.csv"

    comparison_df.to_csv(comparison_file, index=False)

    print("-" * 60)
    print("Model Comparison")
    print("-" * 60)
    print(comparison_df.to_string(index=False))
    print("-" * 60)
    print(f"Saved to: {comparison_file}")

    if len(comparison_df) > 1:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(comparison_df["model"], comparison_df["mae"])
        axes[0].set_title("Test MAE by Model")
        axes[0].set_ylabel("MAE (°C)")
        axes[0].grid(True, axis="y")

        axes[1].bar(comparison_df["model"], comparison_df["rmse"])
        axes[1].set_title("Test RMSE by Model")
        axes[1].set_ylabel("RMSE (°C)")
        axes[1].grid(True, axis="y")

        fig.tight_layout()

        chart_file = OUTPUT_DIR / "comparison_chart.png"

        fig.savefig(chart_file, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved chart to: {chart_file}")


if __name__ == "__main__":
    compare()
