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

# The saved LSTM's evaluation on the same 16-city test split every Transformer
# experiment uses -- a fair per-horizon reference, even though the LSTM model
# itself was originally trained on a separate, larger 82-city archive.
DEFAULT_REFERENCE_METRICS = OUTPUT_DIR / "lstm_on_shared_test_split" / "metrics.json"

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
        "persistence_mae_by_hour": metrics.get("persistence_baseline", {}).get("mae_by_hour"),
        "previous_day_mae_by_hour": metrics.get("previous_day_baseline", {}).get("mae_by_hour"),
        "epochs_trained": epochs_trained
    }


def _load_reference_mae_by_hour(reference_metrics):
    """Reads improved_model.mae_by_hour from an external metrics.json (e.g.
    models/comparison/lstm_on_shared_test_split/metrics.json) plus a label
    built from its model_path, so the overlay can show a model that isn't
    one of `model_names`'s own experiments."""
    with open(reference_metrics) as f:
        metrics = json.load(f)

    mae_by_hour = metrics.get("improved_model", {}).get("mae_by_hour")

    model_path = metrics.get("model_path")
    base_name = Path(model_path).parent.parent.name if model_path else Path(reference_metrics).parent.name

    return mae_by_hour, f"{base_name} (reference -- trained on a separate 82-city archive, not a like-for-like peer)"


def _assert_consistent_baseline(baseline_name, mae_by_hour_by_model):
    """`mae_by_hour_by_model` is {model_name: mae_by_hour}. Every experiment
    being compared scores this baseline on the same test windows, so their
    copies must be numerically identical; a mismatch means the runs weren't
    actually evaluated on the same data, which would silently invalidate
    the comparison."""
    items = list(mae_by_hour_by_model.items())
    if len(items) < 2:
        return

    reference_model, reference_values = items[0]
    for model_name, values in items[1:]:
        if not np.allclose(values, reference_values):
            raise AssertionError(
                f"{baseline_name}.mae_by_hour differs between '{reference_model}' and "
                f"'{model_name}' -- these experiments were not evaluated on the same "
                "test windows, so their metrics cannot be compared."
            )


def compare(model_names=MODEL_NAMES, name=None, reference_metrics=None):
    """`reference_metrics`, if given, is a path to another model's
    metrics.json (e.g. models/comparison/lstm_on_shared_test_split/metrics.json)
    whose improved_model.mae_by_hour is added to the by-horizon overlay as a
    dashed line, labelled as a reference so it isn't mistaken for one of
    `model_names`'s own experiments."""
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

    # Baselines must be identical across every experiment being compared --
    # they're scored on the same test windows -- so this both verifies that
    # assumption and picks a representative copy to plot.
    persistence_by_model = {
        row["model"]: row["persistence_mae_by_hour"] for row in rows if row["persistence_mae_by_hour"]
    }
    previous_day_by_model = {
        row["model"]: row["previous_day_mae_by_hour"] for row in rows if row["previous_day_mae_by_hour"]
    }
    _assert_consistent_baseline("persistence_baseline", persistence_by_model)
    _assert_consistent_baseline("previous_day_baseline", previous_day_by_model)
    persistence_mae_by_hour = next(iter(persistence_by_model.values()), None)
    previous_day_mae_by_hour = next(iter(previous_day_by_model.values()), None)

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

    if mae_by_hour_by_model or persistence_mae_by_hour or previous_day_mae_by_hour or reference_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name, mae_by_hour in mae_by_hour_by_model.items():
            hours = np.arange(1, len(mae_by_hour) + 1)
            ax.plot(hours, mae_by_hour, marker="o", markersize=3, label=model_name)

        # All three reference lines are dashed and muted grey (as opposed to
        # the trained configs' solid, colorful lines), in different shades/
        # styles so they stay distinguishable from each other too.
        if persistence_mae_by_hour:
            hours = np.arange(1, len(persistence_mae_by_hour) + 1)
            ax.plot(hours, persistence_mae_by_hour, linestyle="--", color="0.6",
                     linewidth=1.5, label="Persistence baseline")

        if previous_day_mae_by_hour:
            hours = np.arange(1, len(previous_day_mae_by_hour) + 1)
            ax.plot(hours, previous_day_mae_by_hour, linestyle="--", color="0.3",
                     linewidth=1.5, label="Previous-day baseline")

        if reference_metrics is not None:
            reference_mae_by_hour, reference_label = _load_reference_mae_by_hour(reference_metrics)
            if reference_mae_by_hour:
                hours = np.arange(1, len(reference_mae_by_hour) + 1)
                ax.plot(hours, reference_mae_by_hour, linestyle=":", color="0.45",
                         linewidth=2, label=reference_label)
            else:
                print(f"No improved_model.mae_by_hour found in {reference_metrics}, skipping reference overlay.")

        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("MAE (°C)")
        ax.set_title("MAE by Forecast Horizon")
        ax.set_xlim(left=1)
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        overlay_file = comparison_dir / "mae_by_horizon_overlay.png"

        fig.savefig(overlay_file, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print(f"Saved MAE-by-horizon overlay to: {overlay_file}")

    return comparison_df


def compare_all(reference_metrics=DEFAULT_REFERENCE_METRICS):
    """Compares every model that has at least one experiment with a
    metrics.json, in one combined table + chart under models/comparison/all/.

    `reference_metrics` defaults to the saved LSTM's evaluation on the
    shared 16-city test split (see src.models.common.evaluate_saved_lstm),
    so the by-horizon overlay includes it as a labelled reference. Pass
    None to omit it."""

    if reference_metrics is not None and not Path(reference_metrics).exists():
        print(f"Reference metrics not found at {reference_metrics}, skipping reference overlay.")
        reference_metrics = None

    return compare(discover_model_names(), name="all", reference_metrics=reference_metrics)


if __name__ == "__main__":
    compare_all()
