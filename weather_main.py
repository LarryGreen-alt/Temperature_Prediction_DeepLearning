import os

import subprocess

from src.data.collect_historical import (
    save_historical_weather
)

from src.data.preprocess import (
    preprocess
)

from src.data.split_dataset import (
    split_dataset
)

from src.utils.city_coordinates import (
    CITY_COORDINATES,
    get_coordinates
)

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/experiments", exist_ok=True)


print()

print("==============================")
print("Weather Temperature AI")
print("==============================")

print("\nAvailable Cities\n")

for city in CITY_COORDINATES:

    print(f" - {city.title()}")

print()

while True:

    city_name = input(
        "Enter city: "
    ).lower().strip()

    if city_name in CITY_COORDINATES:
        break

    print("Invalid city.\n")


latitude, longitude = get_coordinates(city_name)

print("\nDownloading weather history...\n")

save_historical_weather(

    latitude=latitude,

    longitude=longitude,

    start_date="2015-01-01",

    end_date="2024-12-31",

    output_file=f"data/raw/{city_name}.csv"

)

print("\nPreprocessing...\n")

preprocess()

print("\nSplitting datasets...\n")

split_dataset()

print()

print("==============================")
print("Dataset Ready")
print("==============================")

print()

print("Run the following next:")

print()

print("python src/models/LSTM.py")

print()

print("After training:")

print()

print("python src/models/model.py")

print("\nAvailable Models\n")
print(" - lstm")
print(" - transformer")
print(" - gru")

while True:

    model_choice = input(
        "Choose model: "
    ).lower().strip()

    if model_choice in ["lstm", "transformer", "gru"]:
        break

    print("Invalid model.\n")

print(f"\nStarting {model_choice.upper()} training...\n")

subprocess.run(
    ["python", "-m", f"src.models.{model_choice}.model"],
    check=True
)