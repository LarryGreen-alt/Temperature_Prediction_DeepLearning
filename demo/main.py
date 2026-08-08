"""FastAPI app for the live forecast demo.

Serves two JSON endpoints (/api/cities, /api/predict) and the static
frontend. Run with:

    uvicorn demo.main:app --reload

then open http://localhost:8000
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from demo import inference, live_weather
from src.models.common.data import CITY_VOCAB, OUTPUT_HOURS

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Live Temperature Forecast Demo")


def _city_label(city_key):
    if city_key == "random":
        return "Random (out-of-distribution probe)"
    return city_key.title()


@app.get("/api/cities")
def get_cities():
    return [
        {"key": city_key, "label": _city_label(city_key)}
        for city_key in sorted(CITY_VOCAB)
    ]


@app.get("/api/predict")
def get_predict(city: str, model: str):
    city_key = city.strip().lower()
    model_name = model.strip().lower()

    if city_key not in CITY_VOCAB:
        raise HTTPException(status_code=400, detail=f"Unknown city '{city}'.")
    if model_name not in ("lstm", "transformer"):
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}', expected 'lstm' or 'transformer'.")

    try:
        features, last_observed = live_weather.get_live_window(city_key)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Failed to fetch live weather data: {error}")

    try:
        predicted_temperatures = inference.predict(model_name, city_key, features)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    last_observed_time = pd.Timestamp(last_observed[-1]["time"])
    forecast_start = last_observed_time + pd.Timedelta(hours=1)
    forecast_times = pd.date_range(start=forecast_start, periods=OUTPUT_HOURS, freq="h")

    forecast = [
        {
            "hour": hour,
            "time": forecast_time.isoformat(),
            "temperature_c": temperature,
        }
        for hour, (forecast_time, temperature) in enumerate(
            zip(forecast_times, predicted_temperatures), start=1
        )
    ]

    return {
        "city": city_key,
        "model": model_name,
        "last_observed": last_observed,
        "forecast": forecast,
    }


# Registered last so it only catches whatever /api/* didn't already claim.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
