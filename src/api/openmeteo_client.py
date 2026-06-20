import requests


def get_historical_weather(
    latitude,
    longitude,
    start_date,
    end_date
):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly="
        "temperature_2m,"
        "relative_humidity_2m,"
        "surface_pressure,"
        "wind_speed_10m,"
        "cloud_cover,"
        "precipitation,"
        "is_day"
    )

    response = requests.get(url)

    response.raise_for_status()

    return response.json()