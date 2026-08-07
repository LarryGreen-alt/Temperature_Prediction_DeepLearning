# Experiment 1: temperature as an input feature

**Hypothesis**: `temperature_2m` is the prediction target, but its own recent
history is also a strong autoregressive signal. Does including the past
`INPUT_HOURS` (72) hours of temperature alongside the existing weather
covariates (humidity, pressure, wind, cloud cover, precipitation, is_day,
time-of-day/day-of-year encodings) improve test MAE/RMSE over the
weather-only [`exp00_baseline`](../exp00_baseline/) — and does that
improvement hold across the full 24-hour forecast, or decay at longer
horizons where the model can no longer lean on recent persistence?

**What changed vs. baseline**: nothing except the input features.
`FEATURE_COLUMNS` in [`config.py`](config.py) is the canonical 11-column
list (`src.models.common.data.FEATURE_COLUMNS`), which already leads with
`temperature_2m` — versus exp00's 10, with it removed. Architecture,
hyperparameters, data splits, and training loop are otherwise identical;
`architecture.build_model()` infers `num_features` from the data at runtime,
so no other code changes.

**How to run**

```
python -m src.models.transformer.experiments.exp01_temperature.train
```

Runs locally on CPU (see `run_all_experiments.py` at the repo root to run
all five experiments in sequence).

**Results**: saved under
`models/Transformer/exp01_temperature/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions_sample.csv`,
`training_history.csv`, `model_summary.txt`, `figures/*.png`). See
`notebooks/results.ipynb` for the cross-experiment narrative write-up.
