# Experiment 4: residual/baseline blending

**Hypothesis**: Larry's LSTM predicts a *correction* to a persistence-like
estimate rather than the raw temperature — its architecture explicitly
blends learned baselines (previous-day, persistence, trend) with a learned
correction term. Its h=3 bias against the pre-realignment Transformer was
+0.07°C vs. −1.67°C. Does giving the Transformer the same residual
formulation — forecast a correction, then add back the last observed
temperature — close that gap, or was the LSTM's advantage really about
recurrence rather than the output formulation?

**What changed vs. [`exp03_temperature_city`](../exp03_temperature_city/)**:
inputs are unchanged (temperature history + city embedding, `CONFIG` here
differs from exp03's only in `baseline_blend=True`); the architecture's head
changes:

- `src/models/transformer/architecture.py`'s `build_model()`, when
  `baseline_blend=True`, adds the last observed raw temperature — read from
  the raw `weather_sequence` input at `[:, -1, temperature_index]`, *before*
  `TimeDistributed(normalizer)`, so the added quantity is in degrees Celsius
  and matches the unscaled target — to every one of the 24 forecast hours.
  The `Dense(output_hours)` head therefore only has to learn the *delta*
  from last-observed, not the absolute temperature.
- This requires confirming the target column is actually one of
  `FEATURE_COLUMNS` (it is here, same as exp03) — `build_model()` raises
  `ValueError` if `baseline_blend=True` without that confirmation, so a
  future experiment can't silently blend in the wrong feature.

**How to run**

```
python -m src.models.transformer.experiments.exp04_baseline_blend.train
```

Runs locally on CPU (see `run_all_experiments.py` at the repo root to run
all five experiments in sequence).

**Results**: saved under
`models/Transformer/exp04_baseline_blend/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions_sample.csv`,
`training_history.csv`, `model_summary.txt`, `figures/*.png`). See
`notebooks/results.ipynb` for the cross-experiment narrative write-up,
including whether `bias_by_hour` moved toward the LSTM's near-zero bias.
