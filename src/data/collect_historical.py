import pandas as pd

from src.api.openmeteo_client import (
    get_historical_weather
)

# Can evolve with getting years of data specific times before and train our model.

def save_historical_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    output_file
):
    data = get_historical_weather(
        latitude,
        longitude,
        start_date,
        end_date
    )

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    df.to_csv(output_file, index=False)

    print(f"Saved {len(df)} records")