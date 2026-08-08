# Experiment 0: baseline (control)

**Hypothesis**: this isn't a hypothesis-driven experiment — it's the control
every other experiment is measured against. A weather-only Transformer
(humidity, pressure, wind, cloud cover, precipitation, is_day, time-of-day/
day-of-year encodings — 10 features, no temperature history, no city
identity) forecasting `OUTPUT_HOURS` (24) hours ahead from an `INPUT_HOURS`
(72) hour window. Without this run, exp01-exp04 have nothing to be an
ablation against.

**What changed vs. baseline**: nothing — this *is* the baseline.
`FEATURE_COLUMNS` in [`config.py`](config.py) is the canonical 11-column
list (`src.models.common.data.FEATURE_COLUMNS`) with `temperature_2m`
removed. `city_aware` and `baseline_blend` are both left at their defaults
(`False`). Architecture, hyperparameters, data splits, and training loop are
whatever `src/models/transformer/training.py`'s `run_training()` and
`src/models/transformer/architecture.py`'s `build_model()` currently do.

**How to run**

```
python -m src.models.transformer.experiments.exp00_baseline.train
```

Runs locally on CPU (see `run_all_experiments.py` at the repo root to run
all five experiments in sequence).

**Results**: saved under
`models/Transformer/exp00_baseline/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions_sample.csv`,
`training_history.csv`, `model_summary.txt`, `figures/*.png`). `metrics.json`
holds `improved_model`/`persistence_baseline`/`previous_day_baseline`
(each with `mae_by_hour`/`rmse_by_hour`/`bias_by_hour`) plus
`test_metrics_by_group` (per-city) and provenance fields
(`num_windows`, `num_cities`, `input_hours`, `output_hours`, `stride`,
`feature_columns`). See `notebooks/results.ipynb` for the cross-experiment
narrative write-up.
