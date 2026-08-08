"""Builds a side-by-side metrics comparison for the report.

Loads one run per entry -- by default each model's most recent run, or a
caller-chosen specific run -- and writes a comparison table + bar chart to
its own subfolder under models/comparison/, named after the entries
compared.

Every current-format metrics.json (both tracks) is nested under an
"improved_model" key holding overall_mae/overall_rmse/mae_by_hour/... (see
src.models.common.evaluation.evaluate_and_save) -- load_row reads that
shape. Older LSTM runs predate this schema (see load_row's old_format
branch) and are handled gracefully rather than rejected.

Entry syntax accepted by compare()'s model_names list:
    "Transformer/exp04_baseline_blend"          -- latest run, label = model name
    "LSTM@2026-08-02_07-36-51"                   -- that specific run, label = model name
    ("LSTM@2026-08-02_07-36-51", "LSTM final")   -- specific run, caller-chosen label
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

# Cross-track material (e.g. the LSTM's evaluation on the shared test split)
# is never pulled in automatically -- pass reference_metrics explicitly.
DEFAULT_REFERENCE_METRICS = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text):
    return text.replace("/", "-").replace("@", "-")


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


def _parse_entry(entry):
    """Normalizes one compare() model_names entry into (model_name,
    experiment_id, label).

    Accepted forms:
        "Transformer/exp04_baseline_blend"          -- latest run, label = model name
        "LSTM@2026-08-02_07-36-51"                   -- that specific run, label = model name
        ("LSTM@2026-08-02_07-36-51", "LSTM final")   -- specific run, caller-chosen label

    experiment_id is None when unspecified, meaning "resolve to the latest
    run with a metrics.json" (see latest_experiment_with_metrics). This is
    what lets multiple entries reference the same model -- e.g. two
    different runs of "LSTM" -- as long as each is given its own label."""
    if isinstance(entry, tuple):
        entry_str, label = entry
    else:
        entry_str, label = entry, None

    model_name, _, experiment_id = entry_str.partition("@")
    experiment_id = experiment_id or None

    if label is None:
        label = model_name

    return model_name, experiment_id, label


def resolve_experiment_dir(model_name, experiment_id=None):
    if experiment_id is None:
        return latest_experiment_with_metrics(model_name)

    experiment_dir = MODEL_DIR / model_name / "experiments" / experiment_id
    return experiment_dir if (experiment_dir / "metrics.json").exists() else None


def load_row(model_name, experiment_id=None, label=None):
    experiment_dir = resolve_experiment_dir(model_name, experiment_id)
    identifier = f"{model_name}@{experiment_id}" if experiment_id else model_name
    label = label or model_name

    if experiment_dir is None:
        print(f"No experiment with metrics.json found for {identifier}, skipping.")
        return None

    with open(experiment_dir / "metrics.json") as f:
        metrics = json.load(f)

    epochs_trained = None

    history_file = experiment_dir / "training_history.csv"

    if history_file.exists():
        epochs_trained = len(pd.read_csv(history_file))

    if "improved_model" in metrics:
        improved_model = metrics["improved_model"]
        return {
            "model": label,
            "experiment": experiment_dir.name,
            "mae": improved_model.get("overall_mae"),
            "rmse": improved_model.get("overall_rmse"),
            "mae_by_hour": improved_model.get("mae_by_hour"),
            "persistence_mae_by_hour": metrics.get("persistence_baseline", {}).get("mae_by_hour"),
            "previous_day_mae_by_hour": metrics.get("previous_day_baseline", {}).get("mae_by_hour"),
            "epochs_trained": epochs_trained,
            "old_format": False,
        }

    # Older LSTM runs predate the improved_model/mae_by_hour schema -- some
    # are a flat {loss, mae, rmse}, others use different top-level key names
    # entirely. Either way there's no per-horizon breakdown to overlay, but
    # whatever scalar mae/rmse exists at the top level still belongs in the
    # summary table and bar chart.
    return {
        "model": label,
        "experiment": experiment_dir.name,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mae_by_hour": None,
        "persistence_mae_by_hour": None,
        "previous_day_mae_by_hour": None,
        "epochs_trained": epochs_trained,
        "old_format": True,
    }


def _parse_reference(reference_metrics):
    """Normalizes compare()'s reference_metrics into (path, label). Accepts
    a bare path (label defaults to its parent directory's name) or a
    (path, label) tuple for a caller-chosen label."""
    if isinstance(reference_metrics, tuple):
        path, label = reference_metrics
    else:
        path, label = reference_metrics, None

    if label is None:
        label = Path(path).parent.name

    return path, label


def _load_reference_mae_by_hour(reference_path):
    """Reads improved_model.mae_by_hour from an external metrics.json (e.g.
    models/comparison/lstm_on_shared_test_split/metrics.json), so the
    overlay can show a run that isn't one of `model_names`'s own entries."""
    with open(reference_path) as f:
        metrics = json.load(f)

    return metrics.get("improved_model", {}).get("mae_by_hour")


def _assert_consistent_baseline(baseline_name, mae_by_hour_by_model):
    """`mae_by_hour_by_model` is {label: mae_by_hour}. Every entry being
    compared scores this baseline on the same test windows, so their
    copies must be numerically identical; a mismatch means the runs weren't
    actually evaluated on the same data."""
    items = list(mae_by_hour_by_model.items())
    if len(items) < 2:
        return

    reference_model, reference_values = items[0]
    for model_name, values in items[1:]:
        if not np.allclose(values, reference_values):
            raise AssertionError(
                f"{baseline_name}.mae_by_hour differs between '{reference_model}' and "
                f"'{model_name}' -- these entries were not evaluated on the same "
                "test windows, so their metrics cannot be compared."
            )


def compare(model_names=MODEL_NAMES, name=None, reference_metrics=None, include_baselines=True):
    """`model_names` is a list of entries -- see the module docstring for
    the accepted "Model", "Model@run_id", and (entry, label) forms.

    `include_baselines`, when True (the default), adds the persistence and
    previous-day baselines to the by-horizon overlay as dashed reference
    lines, asserting first that every entry's copy of each baseline is
    numerically identical (see _assert_consistent_baseline). Pass False to
    omit them.

    `reference_metrics`, if given, is a path to another run's metrics.json
    (e.g. models/comparison/lstm_on_shared_test_split/metrics.json), or a
    (path, label) tuple for a caller-chosen legend label -- otherwise the
    label defaults to the path's parent directory name. Its
    improved_model.mae_by_hour is added to the overlay as a dashed line, so
    the overlay can show a run that isn't one of `model_names`'s own
    entries."""
    if model_names is None:
        model_names = discover_model_names()

    entries = [_parse_entry(entry) for entry in model_names]

    rows = [
        row
        for row in (
            load_row(model_name, experiment_id, label)
            for model_name, experiment_id, label in entries
        )
        if row is not None
    ]

    if not rows:
        raise FileNotFoundError(
            "No experiments with metrics.json found for any entry. "
            "Train the LSTM and Transformer models first."
        )

    mae_by_hour_by_model = {
        row["model"]: row["mae_by_hour"] for row in rows if row["mae_by_hour"]
    }

    skipped_from_overlay = [row["model"] for row in rows if not row["mae_by_hour"]]
    if skipped_from_overlay:
        print(
            "Skipped from the by-horizon overlay (no per-horizon mae_by_hour data, "
            f"likely an old-format run): {', '.join(skipped_from_overlay)}"
        )

    persistence_mae_by_hour = None
    previous_day_mae_by_hour = None

    if include_baselines:
        # Baselines must be identical across every entry being compared --
        # they're scored on the same test windows -- so this both verifies
        # that assumption and picks a representative copy to plot.
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

    comparison_name = name or "_vs_".join(_slugify(label) for _, _, label in entries)
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

        # All reference lines are dashed and muted grey (as opposed to the
        # trained configs' solid, colorful lines), in different shades/
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
            reference_path, reference_label = _parse_reference(reference_metrics)
            reference_mae_by_hour = _load_reference_mae_by_hour(reference_path)
            if reference_mae_by_hour:
                hours = np.arange(1, len(reference_mae_by_hour) + 1)
                ax.plot(hours, reference_mae_by_hour, linestyle=":", color="0.45",
                         linewidth=2, label=reference_label)
            else:
                print(f"No improved_model.mae_by_hour found in {reference_path}, skipping reference overlay.")

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

    `reference_metrics` defaults to None -- cross-track material (e.g. the
    LSTM's evaluation on the shared 16-city test split) is never pulled in
    automatically; pass it explicitly (see compare())."""

    if reference_metrics is not None:
        reference_path, _ = _parse_reference(reference_metrics)
        if not Path(reference_path).exists():
            print(f"Reference metrics not found at {reference_path}, skipping reference overlay.")
            reference_metrics = None

    return compare(discover_model_names(), name="all", reference_metrics=reference_metrics)


if __name__ == "__main__":
    compare_all()
