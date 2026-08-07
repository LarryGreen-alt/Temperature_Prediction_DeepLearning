# Experiment 2: city identity as an input feature

**Hypothesis**: the training data pools hourly weather from 16 cities with
very different climates (Miami vs. Seattle vs. Phoenix, for example), but
[`exp00_baseline`](../exp00_baseline/) is never told which city a given
window came from. Does giving the Transformer an explicit city identity —
via a learned embedding — let it learn city-specific climatology and improve
test MAE/RMSE over the weather-only baseline? This is expected to matter
more at longer horizons, where the model has less to lean on from recent
history and more to gain from knowing the local climate it's forecasting
for.

**What changed vs. baseline**: unlike [`exp01_temperature`](../exp01_temperature/),
this is an architecture change, not just a wider feature vector — city is
categorical, so it needs its own `Embedding` lookup rather than being folded
into the numeric feature columns. `FEATURE_COLUMNS` in [`config.py`](config.py)
is unchanged from exp00 (10 columns, no temperature history); city is
injected as a separate model input instead:

- `src/models/common/data.py`'s `create_multistep_sequences(..., with_city_ids=True)`
  builds each city's windows independently (a window never spans a city
  boundary or a time gap) and returns a parallel array of city ids alongside
  `X`/`y`.
- `src/models/transformer/architecture.py`'s `build_model()` takes an
  optional `num_cities` argument: when set, a second `city_id` input goes
  through an `Embedding`, gets projected to `d_model`, and is added to every
  timestep of the weather projection — the same way positional encoding is
  added. This bypasses the shared `Normalization` layer entirely, since city
  is categorical, not a continuous quantity to z-score.
- `src/models/transformer/training.py`'s `run_training()` takes a
  `city_aware` flag that feeds `[weather, city_id]` into `model.fit`/
  `model.predict` instead of the bare weather array. Passing
  `city_aware=False` (exp00/exp01's default) reproduces the single-input
  code path.

**Design note**: the city embedding is added to the weather projection
rather than concatenated before it. Since that projection is a linear layer,
the two are mathematically equivalent (no expressiveness is lost) — adding
it afterward just avoids routing the categorical id through the same
`Normalization` layer as the continuous weather features.

**How to run**

```
python -m src.models.transformer.experiments.exp02_city.train
```

Runs locally on CPU (see `run_all_experiments.py` at the repo root to run
all five experiments in sequence).

**Results**: saved under
`models/Transformer/exp02_city/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions_sample.csv`,
`training_history.csv`, `model_summary.txt`, `figures/*.png`). See
`notebooks/results.ipynb` for the cross-experiment narrative write-up.
