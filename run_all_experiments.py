"""Runs the five Transformer experiments sequentially, each as its own
subprocess so one crash doesn't take down the rest of the queue.

Order matters -- it's chosen so the run degrades gracefully if interrupted
or time runs out: exp03 and exp00 alone already constitute a valid ablation
(temperature history + city identity vs. neither), so they run first, then
exp01 isolates the temperature effect, exp04 (highest upside per GPU/CPU
minute) comes next, and exp02 -- the one that completes the 2x2 grid -- runs
last and is the first to cut if time runs out.

Experiments that already have a completed run (a readable metrics.json
under their latest experiment directory) are skipped by default, so a
crashed queue is recoverable by re-running the exact same command --
only what's missing gets (re-)trained. Pass --force to re-run everything
requested regardless.

Usage:
    python run_all_experiments.py                          # all five, default order
    python run_all_experiments.py exp00_baseline exp01_temperature   # a subset
    python run_all_experiments.py --force                  # re-run all five anyway
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "Transformer"

DEFAULT_ORDER = [
    "exp03_temperature_city",
    "exp00_baseline",
    "exp01_temperature",
    "exp04_baseline_blend",
    "exp02_city",
]


def latest_experiment_dir(experiment_name):
    experiments_dir = MODELS_DIR / experiment_name / "experiments"
    if not experiments_dir.exists():
        return None
    candidates = sorted((d for d in experiments_dir.iterdir() if d.is_dir()), reverse=True)
    return candidates[0] if candidates else None


def count_epochs(history_file):
    if not history_file.exists():
        return None
    with open(history_file) as f:
        return max(sum(1 for _ in f) - 1, 0)  # rows minus the header


def completed_experiment_dir(experiment_name):
    """latest_experiment_dir(experiment_name) if it exists and has a
    readable metrics.json (i.e. a completed run), else None."""
    experiment_dir = latest_experiment_dir(experiment_name)
    if experiment_dir is None:
        return None
    try:
        with open(experiment_dir / "metrics.json") as f:
            json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return experiment_dir


def row_from_experiment_dir(experiment_name, experiment_dir, status):
    with open(experiment_dir / "metrics.json") as f:
        metrics = json.load(f)
    return {
        "experiment": experiment_name,
        "status": status,
        "start": None,
        "end": None,
        "duration_min": 0.0,
        "epochs_trained": count_epochs(experiment_dir / "training_history.csv"),
        "model_mae": metrics.get("improved_model", {}).get("overall_mae"),
        "persistence_mae": metrics.get("persistence_baseline", {}).get("overall_mae"),
    }


def run_experiment(experiment_name, force=False):
    if not force:
        experiment_dir = completed_experiment_dir(experiment_name)
        if experiment_dir is not None:
            row = row_from_experiment_dir(experiment_name, experiment_dir, status="skipped")
            print(f"{experiment_name}: already complete (model MAE {row['model_mae']}) -- "
                  f"skipping. Pass --force to re-run.")
            return row

    module = f"src.models.transformer.experiments.{experiment_name}.train"
    start = datetime.now()
    print(f"\n{'=' * 70}\n{experiment_name}: starting at {start.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}")

    result = subprocess.run([sys.executable, "-m", module], cwd=PROJECT_ROOT)
    end = datetime.now()

    row = {
        "experiment": experiment_name,
        "status": "ok" if result.returncode == 0 else f"failed ({result.returncode})",
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round((end - start).total_seconds() / 60, 1),
        "epochs_trained": None,
        "model_mae": None,
        "persistence_mae": None,
    }

    if result.returncode != 0:
        print(f"{experiment_name}: FAILED (exit code {result.returncode})")
        return row

    experiment_dir = latest_experiment_dir(experiment_name)
    if experiment_dir is None:
        print(f"{experiment_name}: completed but no experiment directory was found under {MODELS_DIR}")
        return row

    try:
        with open(experiment_dir / "metrics.json") as f:
            metrics = json.load(f)
        row["model_mae"] = metrics["improved_model"]["overall_mae"]
        row["persistence_mae"] = metrics["persistence_baseline"]["overall_mae"]
    except (FileNotFoundError, KeyError) as exc:
        print(f"{experiment_name}: could not read metrics.json ({exc})")

    row["epochs_trained"] = count_epochs(experiment_dir / "training_history.csv")

    print(
        f"{experiment_name}: done in {row['duration_min']:.1f} min, "
        f"{row['epochs_trained']} epochs, "
        f"model MAE {row['model_mae']}, persistence MAE {row['persistence_mae']}"
    )
    return row


def print_summary(rows):
    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    header = f"{'experiment':<26}{'status':<12}{'epochs':<8}{'model MAE':<12}{'persist. MAE':<14}{'duration (min)'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        model_mae = f"{row['model_mae']:.4f}" if row["model_mae"] is not None else "-"
        persistence_mae = f"{row['persistence_mae']:.4f}" if row["persistence_mae"] is not None else "-"
        epochs = row["epochs_trained"] if row["epochs_trained"] is not None else "-"
        print(f"{row['experiment']:<26}{row['status']:<12}{str(epochs):<8}{model_mae:<12}{persistence_mae:<14}{row['duration_min']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiments", nargs="*", default=None,
        help="Subset of experiment names to run, in the order given. "
             f"Defaults to all five in degrade-gracefully order: {', '.join(DEFAULT_ORDER)}."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every requested experiment even if it already has a completed run "
             "(default: skip experiments that already have a readable metrics.json)."
    )
    args = parser.parse_args()

    experiments = args.experiments if args.experiments else DEFAULT_ORDER

    rows = [run_experiment(name, force=args.force) for name in experiments]
    print_summary(rows)

    if any(row["status"].startswith("failed") for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
