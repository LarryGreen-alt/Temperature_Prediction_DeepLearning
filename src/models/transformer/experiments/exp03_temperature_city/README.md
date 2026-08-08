# Experiment 3: combining temperature history and city identity

**Hypothesis**: [`exp01_temperature`](../exp01_temperature/) (the target's
own recent history as an input) and [`exp02_city`](../exp02_city/) (a
learned city identity embedding) each independently change test MAE/RMSE
versus [`exp00_baseline`](../exp00_baseline/). Are their effects additive —
does combining both give a model at least as good as the better of the two
on its own — or do they interact (positively or negatively) when applied
together? This is also the configuration most directly comparable to the
LSTM track (`models/LSTM/`), which always has both temperature history and
(implicitly, via architecture) locality.

**What changed vs. baseline**: both changes are applied at once, unmodified
from how each experiment introduced them:

- `FEATURE_COLUMNS` in [`config.py`](config.py) is exp01's list — the
  canonical 11 columns, temperature history included.
- `train.py` calls `run_training(..., city_aware=True, city_embed_dim=CITY_EMBED_DIM)`,
  exp02's change — city identity injected as a second model input.

No new code was needed: `run_training()` already accepts an arbitrary
`feature_columns` list alongside `city_aware`/`city_embed_dim` independently
of each other, since neither exp01 nor exp02 assumed the other was off.
This experiment is the first to set both at the same time.

**How to run**

```
python -m src.models.transformer.experiments.exp03_temperature_city.train
```

Runs locally on CPU (see `run_all_experiments.py` at the repo root to run
all five experiments in sequence — this one first, since it's both the
strongest candidate and the one most comparable to the LSTM).

**Results**: saved under
`models/Transformer/exp03_temperature_city/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions_sample.csv`,
`training_history.csv`, `model_summary.txt`, `figures/*.png`). See
`notebooks/results.ipynb` for the cross-experiment narrative write-up,
including the comparison against `models/comparison/lstm_on_shared_test_split/metrics.json`.
