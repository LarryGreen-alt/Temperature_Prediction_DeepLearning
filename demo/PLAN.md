# Live forecast demo — build plan

## Goal

A local demo, FastAPI backend + a single vanilla-JS HTML page, for the video
recording. User picks a city (dropdown, 16 options) and a model (LSTM or
Transformer), hits Predict, and sees a 24-hour forecast built from live
weather data pulled from Open-Meteo at request time.

**This is a demo, not a product.** No auth, no tests, no deployment, no
build step, no error-recovery infrastructure. It needs to work reliably for
one recording session today, nothing more. Prefer the simplest thing that
works over anything "correct" in a production sense.

## Non-negotiable technical facts (verified against this repo, don't re-derive these)

- **Both models expect the same 11 input features, in this exact order:**
  `temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
  cloud_cover, precipitation, is_day, hour_sin, hour_cos, day_sin, day_cos`.
  This list already exists as `FEATURE_COLUMNS` in
  `src/models/common/data.py` — import it, don't retype it.
- **Input window is 72 hours, output is 24 hours**, both hourly. Also already
  constants: `INPUT_HOURS`, `OUTPUT_HOURS` in `src/models/common/data.py`.
- **The LSTM has custom Keras layers** (`TemperatureBaselines`,
  `FutureCalendarFeatures`, `HorizonEmbedding`, `WeightedBaseline`,
  `LevelChangeHuber`). Loading it with a bare `tf.keras.models.load_model()`
  will fail or silently misbehave. **Always load it with
  `load_weather_model()` from `src/models/lstm/model.py`** — that function
  already passes the right `custom_objects`.
- **The LSTM takes a single input** (the 72×11 weather window). It does
  **not** take a city id — city identity isn't part of its architecture at
  all, it only affects which real-world city's data fills the window.
- **The Transformer (exp04, the one we're demoing) takes two inputs**:
  the 72×11 weather window, and a city id, `[X, city_id]` in that order.
  City id is an integer index, not the city name. The mapping is
  `CITY_VOCAB` in `src/models/common/data.py`,
  `{city: idx for idx, city in enumerate(sorted(CITY_COORDINATES))}`.
  **Do not reimplement this mapping by hand or change the sort order** —
  the model's embedding weights are trained against this exact ordering,
  and a different ordering will silently produce wrong predictions for
  every city except the ones that happen to sort first. Import `CITY_VOCAB`
  directly.
- **City coordinates** already exist: `CITY_COORDINATES` in
  `src/utils/city_coordinates.py`, 16 lowercase keys, `(lat, lon)` tuples.
  One entry, `"random"`, is an out-of-distribution probe that resolves to a
  point in North Korea, not a real city. For the dropdown label, show it as
  something honest like "Random (out-of-distribution probe)" rather than
  hiding it, but **do not rename the dict key** `"random"` itself — that
  key is what `CITY_VOCAB` was built from, and renaming it would remap
  every city to the wrong embedding index.
- **Model paths to load (current best of each):**
  - Transformer: `models/Transformer/exp04_baseline_blend/checkpoints/best.keras`
  - LSTM: `models/LSTM/checkpoints/best.keras`
  Both exist right now. Load with plain `tf.keras.models.load_model(path)`
  for the Transformer, and `load_weather_model(path)` for the LSTM (see
  above).
- **Feature engineering for live data** (turning raw Open-Meteo hourly rows
  into the 11 columns) is a small, already-solved problem — mirror the
  cyclical-encoding logic in `src/data/preprocess.py`:
  `hour_sin = sin(2*pi*hour/24)`, `hour_cos = cos(2*pi*hour/24)`,
  `day_sin = sin(2*pi*day_of_year/365)`, `day_cos = cos(2*pi*day_of_year/365)`.
  Don't touch normalization — it's baked into the saved model graph, raw
  values in their natural units are exactly what the model expects.

## The one real gotcha: which Open-Meteo endpoint

Training data came from the **archive** endpoint
(`archive-api.open-meteo.com/v1/archive`, see
`src/api/openmeteo_client.py`), which serves ERA5 reanalysis. Reanalysis is
not available in real time, it lags by several days, so calling the archive
endpoint for "the last 72 hours up to now" will return incomplete or
missing recent data.

**Use the forecast endpoint instead for live data:**
`https://api.open-meteo.com/v1/forecast`, with `latitude`, `longitude`,
`past_hours=72`, and the same `hourly=` variable list as the archive client
(`temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover,precipitation,is_day`).
This serves recent observations/short-range analysis, not reanalysis —
slightly different data source than training saw. That's fine, and honestly
worth a one-line mention on the page itself ("live data, not the reanalysis
product the models trained on") rather than treating it as a bug to hide.

## Suggested structure

```
demo/
  PLAN.md              <- this file
  main.py              <- FastAPI app: serves the API and the static frontend
  inference.py         <- loads both models once at startup, runs predictions
  live_weather.py      <- Open-Meteo forecast fetch + feature engineering
  static/
    index.html          <- the whole frontend, vanilla JS, no build step
  requirements.txt      <- just the two new deps: fastapi, uvicorn
                           (tensorflow/pandas/numpy/requests already exist
                           in the repo's environment)
  README.md             <- one command to run it, e.g.
                           `uvicorn demo.main:app --reload` then open
                           http://localhost:8000
```

## API

Two endpoints, both `GET`, both simple enough to test with a browser URL
bar:

- `GET /api/cities` → list of `{key, label}` for the dropdown, `key` being
  the raw lowercase city name (`"new york"`), `label` a display-friendly
  version (`"New York"`, and `"Random (out-of-distribution probe)"` for
  `"random"`).
- `GET /api/predict?city=<key>&model=<lstm|transformer>` → fetches live
  data for that city, runs the chosen model, returns JSON:
  ```json
  {
    "city": "new york",
    "model": "transformer",
    "last_observed": [{"time": "...", "temperature_c": 21.3}, ...],
    "forecast": [{"hour": 1, "time": "...", "temperature_c": 20.8}, ...]
  }
  ```
  `last_observed` is the 72-hour input window (useful for the chart to show
  continuity between observed and predicted), `forecast` is the 24
  predicted hours.

Load both models **once at startup**, not per-request — TensorFlow model
loading is slow enough that doing it per click would make the demo feel
broken.

## Frontend

One `index.html`, vanilla JS, `fetch()` to the two endpoints above. No
framework, no npm, no build step — FastAPI can serve this file directly as
a static file, so the whole demo is one process, one command to start.

Minimum viable UI:
- City `<select>`, populated from `/api/cities` on page load
- Model `<select>` with two options, LSTM / Transformer
- A "Predict" button
- A chart showing the 72-hour observed line flowing into the 24-hour
  forecast line, so it visually reads as "past becomes future" — a line
  chart via Chart.js from a CDN `<script>` tag is the fastest way to get
  this without a build step. A plain table of the 24 forecast values is an
  acceptable fallback if the chart is taking too long.

## Recommended for reliability during the actual recording (optional, but cheap)

Cache each city's live-fetch response in memory for ~30 minutes after first
fetch. This isn't for correctness, it's so that if Open-Meteo hiccups or
rate-limits mid-recording, a repeated click on a city already demoed in
this session still works instantly instead of depending on a second live
network call. A plain in-process dict keyed by city name is enough — no
database, no persistence needed.

## Explicitly out of scope

Don't build: authentication, request logging, automated tests, a build
pipeline, Docker, deployment config, retry/backoff logic beyond a basic
try/except with a readable error message, support for choosing a custom
date range, or any UI polish beyond "looks intentional in a recorded
video." If something here would take more than a few minutes, it's
probably out of scope for today.
