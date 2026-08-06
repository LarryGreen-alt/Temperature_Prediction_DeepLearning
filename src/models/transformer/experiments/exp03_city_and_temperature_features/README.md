# Experiment 3: combining temperature history and city identity

**Hypothesis**: [`exp01_temperature_feature`](../exp01_temperature_feature/)
(the target's own recent history as an input) and
[`exp02_city_embedding`](../exp02_city_embedding/) (a learned city identity
embedding) each independently changed test MAE/RMSE versus the weather-only
baseline. Are their effects additive -- does combining both give a model
that's at least as good as the better of the two on its own, or do they
interact (positively or negatively) when applied together?

**What changed vs. baseline**: both changes are applied at once, unmodified
from how each experiment introduced them:

- `FEATURE_COLUMNS` in [`config.py`](config.py) is exp01's list --
  baseline features plus `temperature_2m` (the target's own past
  `WINDOW_SIZE` hours).
- `train.py` calls `run_training(..., city_aware=True, city_embed_dim=CITY_EMBED_DIM)`,
  exp02's change -- city identity is injected as a second model input (a
  learned `Embedding`, added to every timestep of the weather projection),
  not as a numeric feature column.

No new code was needed: `src/models/transformer/training.py`'s
`run_training()` already accepts an arbitrary `feature_columns` list
alongside `city_aware`/`city_embed_dim` independently of each other, since
neither exp01 nor exp02 assumed the other was off. This experiment is the
first to set both at the same time.

**How to run**

Locally (smoke test, CPU is fine for a quick sanity check):
```
python -m src.models.transformer.experiments.exp03_city_and_temperature_features.train
```

On Colab (full GPU training): open [`experiment.ipynb`](experiment.ipynb) in
Colab and run all cells -- see the notebook's first cell for one-time setup
(a GitHub PAT stored in Colab's Secrets manager).

**Results**: saved under
`models/Transformer/exp03_city_and_temperature_features/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions.csv`, `training_history.csv`,
`model_summary.txt`, `figures/*.png`) -- see the executed `experiment.ipynb`
for the narrative write-up and inline plots.
