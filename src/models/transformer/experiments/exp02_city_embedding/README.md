# Experiment 2: city identity as an input feature

**Hypothesis**: the training data pools hourly weather from 16 cities with
very different climates (Miami vs. Seattle vs. Phoenix, for example), but the
model is never told which city a given window came from. Does giving the
Transformer an explicit city identity -- via a learned embedding -- let it
learn city-specific patterns and improve test MAE/RMSE over the weather-only
baseline?

**What changed vs. baseline**: unlike [`exp01_temperature_feature`](../exp01_temperature_feature/),
this is an actual architecture change, not just a wider feature vector --
city is categorical, so it needs its own `Embedding` lookup rather than being
folded into the existing numeric feature columns. `FEATURE_COLUMNS` in
[`config.py`](config.py) is unchanged from baseline; city is injected as a
separate model input instead:

- `src/models/common/data.py` adds `CITY_VOCAB` (a city name -> integer id
  mapping built from `src/utils/city_coordinates.py`) and
  `create_city_aware_sequences()`, which builds each city's sliding windows
  independently (a window never spans two cities) and returns a parallel
  array of city ids alongside the usual `X`/`y`.
- `src/models/transformer/architecture.py`'s `build_model()` takes an
  optional `num_cities` argument: when set, a second `city_id` input goes
  through an `Embedding`, gets projected to `d_model`, and is added to every
  timestep of the weather projection -- the same way positional encoding is
  added today. This bypasses the shared `Normalization` layer entirely,
  since city is categorical, not a continuous quantity to z-score.
- `src/models/transformer/training.py`'s `run_training()` takes a
  `city_aware` flag that switches over to the city-aware sequence builder
  and feeds `[weather, city_id]` into `model.fit`/`evaluate`. Passing
  `city_aware=False` (the default) reproduces the exact baseline/exp01 code
  path -- this experiment is the only caller that sets it to `True`.

**Design note**: the city embedding is added to the weather projection
rather than concatenated before it. Since that projection is a linear layer,
the two are mathematically equivalent (no expressiveness is lost) -- adding
it afterward just avoids routing the categorical id through the same
`Normalization` layer as the continuous weather features. A more
"Transformer-native" alternative (prepending the city embedding as an extra
sequence token for attention to use) was considered and rejected: this model
pools with `GlobalAveragePooling1D`, which would dilute a prepended token
1-in-25 and require the (shallow, 2-layer) network to learn to route it to
the other tokens via attention. Adding it directly to every token instead
guarantees the signal survives pooling.

**Side effect**: because a window can no longer span a city boundary, this
experiment trains on a small number of fewer windows than baseline/exp01
(~390 out of ~1.13M, one `window_size + forecast_horizon - 1`-sized gap per
city boundary) -- an unavoidable consequence of tagging each window with a
single city, not an independent change.

**Bug fix included**: `src/data/preprocess.py` derived `city` from the
filename via `.split("_")[0]`, which mangled multi-word cities
(`fort_worth.csv` -> `"fort"`) and left the `.csv` suffix on single-word
cities (`atlanta.csv` -> `"atlanta.csv"`). Fixed to use the full filename
stem so city names match `CITY_COORDINATES`'s keys exactly. This column was
unused by any model before this experiment, so the fix doesn't change any
existing experiment's numeric results -- but required regenerating
`data/processed/features.csv` and `data/splits/*.csv`.

**For a future LSTM comparison**: `CITY_VOCAB`/`create_city_aware_sequences()`
live in `src/models/common/data.py`, shared by both architectures, so an
LSTM city-embedding experiment can reuse the exact same city ids and
windowing for a fair, apples-to-apples comparison -- only the model graph
(its own `Embedding` + second input) needs architecture-specific code.

**How to run**

Locally (smoke test, CPU is fine for a quick sanity check):
```
python -m src.models.transformer.experiments.exp02_city_embedding.train
```

On Colab (full GPU training): open [`experiment.ipynb`](experiment.ipynb) in
Colab and run all cells -- see the notebook's first cell for one-time setup
(a GitHub PAT stored in Colab's Secrets manager).

**Results**: saved under
`models/Transformer/exp02_city_embedding/experiments/<timestamp>/`
(`config.json`, `metrics.json`, `predictions.csv`, `training_history.csv`,
`model_summary.txt`, `figures/*.png`) -- see the executed `experiment.ipynb`
for the narrative write-up and inline plots.
