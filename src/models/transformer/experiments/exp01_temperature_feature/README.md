# Experiment 1: temperature as an input feature

**Hypothesis**: `temperature_2m` is the prediction target, but its own recent
history is also a strong autoregressive signal. Does including the past
`WINDOW_SIZE` hours of temperature alongside the existing weather covariates
(humidity, pressure, wind, cloud cover, precipitation, is_day, time-of-day/
day-of-year encodings) improve test MAE/RMSE over the weather-only baseline?

**What changed vs. baseline**: nothing except the input features.
`FEATURE_COLUMNS` in [`config.py`](config.py) is the baseline feature list
plus `temperature_2m`. Model architecture, hyperparameters, data splits, and
training loop are all identical to `src/models/transformer/model.py` — the
model automatically has one extra input channel since
`architecture.build_model()` infers `num_features` from the data at runtime.

**How to run**

Locally (smoke test, CPU is fine for a quick sanity check):
```
python -m src.models.transformer.experiments.exp01_temperature_feature.train
```

On Colab (full GPU training): open [`experiment.ipynb`](experiment.ipynb) in
Colab and run all cells — see the notebook's first cell for one-time setup
(a GitHub PAT stored in Colab's Secrets manager).

**Results**: saved under
`models/Transformer/exp01_temperature_feature/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions.csv`, `training_history.csv`,
`figures/*.png`) — see the executed `experiment.ipynb` for the narrative
write-up and inline plots.
