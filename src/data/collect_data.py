import pandas as pd
from src.api.weatherapi_client import get_forecast
from src.utils.json_explorer import (
    show_features,
    categorize_features
)
import json


def save_forecast(city):
    data = get_forecast(city)

    show_features(data)

    categorize_features(data)

    records = []

    for day in data["forecast"]["forecastday"]:
        records.append({
            "date": day["date"],
            "avg_temp": day["day"]["avgtemp_f"],
            "humidity": day["day"]["avghumidity"],
            "rain": day["day"]["daily_will_it_rain"],
            "maxwind_mph": day["day"]["maxwind_mph"]
        })

    df = pd.DataFrame(records)

    df.to_csv(
        "data/raw/weather.csv",
        index=False
    )