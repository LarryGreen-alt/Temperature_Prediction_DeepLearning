# Temperature Prediction Deep Learning

This project compares two deep learning architectures for multi-horizon temperature
forecasting: an **LSTM** and a **Transformer encoder**, both built with
TensorFlow/Keras. Given the previous 72 hours of hourly weather observations across
16 U.S. cities, each model forecasts the next 24 hours of temperature.

Both tracks are trained and evaluated on the exact same data splits and windowing, and
scored with the same evaluation code, so their metrics are directly comparable. The
windowing itself is duplicated rather than shared: the LSTM builds its windows through
its own implementation in `src/models/lstm/train.py`, while everything else (the
Transformer experiments, the LSTM evaluation script, and `run_all_experiments.py`) uses
`src/models/common/data.py`. `tests/test_data.py` verifies by direct comparison that
the two implementations produce byte-identical windows from the same input, which is
the basis for treating the two tracks as comparable at all.

## Installation

```
git clone <repo-url>
cd Temperature_Prediction_DeepLearning
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```
Temperature_Prediction_DeepLearning/
├── data/
│   ├── raw/                    # Per-city hourly weather history CSVs
│   └── splits/                 # train.csv / dev.csv / test.csv (generated, git-ignored)
│
├── models/
│   ├── LSTM/
│   │   ├── latest/              # weather_lstm.keras + model_config.json for inference
│   │   └── experiments/<timestamp>/
│   ├── Transformer/
│   │   ├── exp00_baseline/      # weather covariates only
│   │   ├── exp01_temperature/   # + the target's own recent history
│   │   ├── exp02_city/          # + a learned city-identity embedding
│   │   ├── exp03_temperature_city/  # both of the above together
│   │   └── exp04_baseline_blend/    # exp03's inputs + a persistence-residual head
│   │       └── */experiments/<timestamp>/
│   │           ├── weather_transformer.keras
│   │           ├── config.json, metrics.json, model_summary.txt
│   │           ├── training_history.csv
│   │           └── figures/
│   └── comparison/
│       ├── lstm_on_shared_test_split/   # LSTM evaluated on the shared 16-city test split
│       └── h3_phase_results.json        # preserved numbers from the project's earlier
│                                         # single-value, 3-hour-ahead framing (not
│                                         # comparable to the current 72h->24h results)
│
├── notebooks/
│   └── results.ipynb            # loads every experiment's metrics.json and narrates
│                                 # the five hypotheses exp00-exp04 are testing
│
├── src/
│   ├── api/openmeteo_client.py           # Open-Meteo historical weather API client
│   ├── data/
│   │   ├── collect_historical.py         # downloads per-city raw CSVs
│   │   ├── preprocess.py                 # feature engineering -> data/processed/
│   │   └── split_dataset.py              # chronological train/dev/test split
│   │
│   ├── models/
│   │   ├── common/
│   │   │   ├── data.py                   # canonical windowing, shared by everything
│   │   │   │                             # except lstm/train.py's own implementation
│   │   │   ├── evaluation.py             # metrics + persistence/previous-day baselines
│   │   │   ├── plotting.py               # loss/MAE curves, forecast examples
│   │   │   ├── evaluate_saved_lstm.py    # scores the saved LSTM on the shared test split
│   │   │   └── compare.py                # cross-model MAE/RMSE table + MAE-by-horizon overlay
│   │   │
│   │   ├── lstm/
│   │   │   ├── model.py, train.py, predict.py
│   │   │
│   │   └── transformer/
│   │       ├── architecture.py           # build_model(): encoder blocks, city embedding,
│   │       │                             # optional baseline-blend residual head
│   │       ├── training.py               # run_training()/load_cached_results()
│   │       ├── predict.py                # forecast from one experiment's latest model
│   │       └── experiments/
│   │           └── exp*/config.py, train.py, README.md
│   │
│   └── utils/
│       └── city_coordinates.py
│
├── tests/
│   └── test_data.py             # windowing shape/alignment/segmentation tests, plus
│                                 # the byte-identical-to-lstm/train.py cross-check
│
└── run_all_experiments.py       # trains all five Transformer experiments in sequence
```

## Data preparation

```
python -m src.data.collect_historical
python -m src.data.preprocess
python -m src.data.split_dataset
```

## Training

Both tracks train locally on CPU; there is no GPU/Colab dependency.

LSTM:

```
python -m src.models.lstm.train
```

All five Transformer experiments, in degrade-gracefully order (already-completed
experiments are skipped automatically; pass `--force` to retrain everything):

```
python run_all_experiments.py
python run_all_experiments.py exp00_baseline exp01_temperature   # a subset
```

Each Transformer experiment lives in its own folder under
`src/models/transformer/experiments/` with a `config.py` (feature set,
`TransformerConfig` hyperparameters) and a `train.py` entry point, and can also be run
directly:

```
python -m src.models.transformer.experiments.exp03_temperature_city.train
```

## Evaluation and comparison

Score the saved LSTM on the shared 16-city test split:

```
python -m src.models.common.evaluate_saved_lstm
```

Every experiment (both tracks) writes a `metrics.json` nested as
`improved_model` / `persistence_baseline` / `previous_day_baseline`, each holding
`overall_mae`, `overall_rmse`, per-horizon `mae_by_hour`, and a per-city
`test_metrics_by_group` breakdown. Build a side-by-side comparison table plus an
MAE-by-horizon overlay chart across every trained model:

```
python -m src.models.common.compare
```

For a specific subset of runs, call `compare()` directly with a list of entries.
Each entry is either `"Model/name"` (latest run, labelled with the model name),
`"Model/name@<experiment-timestamp>"` (that specific run), or a
`(entry, "custom label")` tuple -- which is how to chart several runs of the same
model side by side, since each needs its own label:

```python
from src.models.common.compare import compare

compare(
    ["Transformer/exp00_baseline", "Transformer/exp01_temperature",
     "Transformer/exp02_city", "Transformer/exp03_temperature_city",
     "Transformer/exp04_baseline_blend"],
    name="transformer_ablation",
)

# Two of Larry's own LSTM runs, explicitly labelled:
compare(
    [("LSTM@2026-08-02_07-36-51", "LSTM (multi-horizon)"),
     ("LSTM@2026-07-18_14-32-02", "LSTM (h=3 era)")],
    name="lstm_runs",
    reference_metrics="models/comparison/lstm_on_shared_test_split/metrics.json",
)
```

`reference_metrics` overlays one more run (a path, or a `(path, label)` tuple)
without it counting as one of the compared entries -- e.g. a cross-track model
evaluated on the same test windows. `compare_all()` (the `python -m
src.models.common.compare` default) never adds a reference automatically; pass
one explicitly when cross-track material belongs in the chart.

`notebooks/results.ipynb` walks through the five Transformer experiments' results in
more detail, alongside the LSTM comparison.

## Making predictions

LSTM (downloads recent observations for a city and forecasts up to 24 hours ahead):

```
python -m src.models.lstm.predict --city seattle
```

Transformer (forecasts from one experiment's latest saved model):

```
python -m src.models.transformer.predict --experiment exp03_temperature_city --city seattle
```

## Tests

```
python -m unittest discover -s tests -t .
```

This should report 7 tests, OK. Among other things, it verifies that
`src/models/common/data.py`'s vectorized windowing produces byte-identical output to a
reimplementation of the LSTM track's own windowing loop
(`src/models/lstm/train.py:344-349`) — the basis for treating the two tracks'
metrics as comparable at all.
