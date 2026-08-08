"""Builds a side-by-side LSTM vs Transformer metrics comparison for the report.

Loads each model's most recent experiment (the one with the highest timestamp
that contains a metrics.json), writes a comparison table + bar chart to its
own subfolder under models/comparison/, named after the models compared.

Every metrics.json in this project (both tracks) is nested under an
"improved_model" key holding overall_mae/overall_rmse/mae_by_hour/... (see
src.models.common.evaluation.evaluate_and_save) -- load_row reads that
shape, not a flat one.
"""

import json

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models"

# None means "every model discovered under models/" (see discover_model_names).
MODEL_NAMES = None

OUTPUT_DIR = MODEL_DIR / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(model_name):
    return model_name.replace("/", "-")


def discover_model_names():
    """Finds every model with at least one experiment that has a
    metrics.json, by looking for models/<model_name>/experiments/ dirs
    anywhere under models/ (model_name may be a single path segment, e.g.
    "LSTM", or nested, e.g. "Transformer/baseline")."""

    names = []

    for experiments_dir in MODEL_DIR.rglob("experiments"):
        if not experiments_dir.is_dir():
            continue

        model_name = experiments_dir.parent.relative_to(MODEL_DIR).as_posix()

        if latest_experiment_with_metrics(model_name) is not None:
            names.append(model_name)

    return sorted(names)


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

    improved_model = metrics.get("improved_model", {})

    epochs_trained = None

    history_file = experiment_dir / "training_history.csv"

    if history_file.exists():
        epochs_trained = len(pd.read_csv(history_file))

    return {
        "model": model_name,
        "experiment": experiment_dir.name,
        "mae": improved_model.get("overall_mae"),
        "rmse": improved_model.get("overall_rmse"),
        "mae_by_hour": improved_model.get("mae_by_hour"),
        "epochs_trained": epochs_trained
    }


def compare(model_names=MODEL_NAMES, name=None):
    if model_names is None:
        model_names = discover_model_names()

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

    mae_by_hour_by_model = {
        row["model"]: row["mae_by_hour"] for row in rows if row["mae_by_hour"]
    }
    table_columns = ["model", "experiment", "mae", "rmse", "epochs_trained"]
    comparison_df = pd.DataFrame(rows, columns=table_columns)

    comparison_name = name or "_vs_".join(_slugify(m) for m in model_names)
    comparison_dir = OUTPUT_DIR / comparison_name
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = comparison_dir / "comparison.csv"

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

        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        fig.tight_layout()

        chart_file = comparison_dir / "comparison_chart.png"

        fig.savefig(chart_file, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved chart to: {chart_file}")

    if mae_by_hour_by_model:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name, mae_by_hour in mae_by_hour_by_model.items():
            hours = np.arange(1, len(mae_by_hour) + 1)
            ax.plot(hours, mae_by_hour, marker="o", markersize=3, label=model_name)

        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("MAE (°C)")
        ax.set_title("MAE by Forecast Horizon")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        overlay_file = comparison_dir / "mae_by_horizon_overlay.png"

        fig.savefig(overlay_file, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved MAE-by-horizon overlay to: {overlay_file}")

    return comparison_df


def compare_all():
    """Compares every model that has at least one experiment with a
    metrics.json, in one combined table + chart under models/comparison/all/."""

    return compare(discover_model_names(), name="all")


if __name__ == "__main__":
    compare_all()
