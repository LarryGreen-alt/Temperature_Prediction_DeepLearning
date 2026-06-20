import requests
from src.utils.config import WEATHER_API_KEY


def get_current_weather(city):
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={WEATHER_API_KEY}"
        f"&q={city}"
    )

    response = requests.get(url)

    return response.json()


def get_forecast(city, days=7):
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={WEATHER_API_KEY}"
        f"&q={city}"
        f"&days={days}"
    )

    response = requests.get(url)

    return response.json()


def get_historical_weather(city, date):
    url = (
        f"http://api.weatherapi.com/v1/history.json"
        f"?key={WEATHER_API_KEY}"
        f"&q={city}"
        f"&dt={date}"
    )

    response = requests.get(url)

    return response.json()