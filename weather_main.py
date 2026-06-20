from src.data.collect_historical import (
    save_historical_weather
)

from src.utils.city_coordinates import (
    get_coordinates,
    CITY_COORDINATES 
)

print("\nAvailable Cities:")

for city in CITY_COORDINATES:
    print(f" - {city.title()}")

while True:
    city_name = input("\nEnter city: ").lower().strip()

    if city_name in CITY_COORDINATES:
        break

    print(f"\n'{city_name}' is not available.")
    print("Please choose from the list above.")

latitude, longitude = get_coordinates(city)

save_historical_weather(
    latitude=latitude,
    longitude=longitude,
    start_date="2024-01-01",
    end_date="2024-12-31",
    output_file=f"data/raw/{city_name.lower()}_2024.csv"
)