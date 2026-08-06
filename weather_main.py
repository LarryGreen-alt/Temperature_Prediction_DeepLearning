from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from src.data.collect_historical import save_historical_weather
from src.data.preprocess import preprocess
from src.data.split_dataset import split_dataset
from src.utils.city_coordinates import CITY_COORDINATES, get_coordinates


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_MODULES = {
    "lstm": "src.models.lstm.train",
    "transformer": "src.models.transformer.train",
    "gru": "src.models.gru.train",
}


def create_project_directories() -> None:
    """Create the shared directories used by the weather pipeline."""
    for directory in [RAW_DIR, PROCESSED_DIR, SPLIT_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def choose_city() -> str:
    """Prompt until the user selects a configured city."""
    print("\nAvailable Cities\n")
    for city in CITY_COORDINATES:
        print(f" - {city.title()}")

    while True:
        city_name = input("\nEnter city: ").strip().lower()
        if city_name in CITY_COORDINATES:
            return city_name
        print("Invalid city. Choose one of the cities shown above.")


def available_models() -> dict[str, str]:
    """Return only models that currently provide a train.py module."""
    available: dict[str, str] = {}

    for model_name, module_name in MODEL_MODULES.items():
        train_path = (
            PROJECT_ROOT
            / "src"
            / "models"
            / model_name
            / "train.py"
        )
        if train_path.exists():
            available[model_name] = module_name

    return available


def choose_model(models: dict[str, str]) -> str:
    """Prompt until the user selects an available training module."""
    if not models:
        raise FileNotFoundError(
            "No model training modules were found. Expected a file such as "
            "src/models/lstm/train.py."
        )

    print("\nAvailable Models\n")
    for model_name in models:
        print(f" - {model_name}")

    while True:
        model_choice = input("\nChoose model: ").strip().lower()
        if model_choice in models:
            return model_choice
        print("Invalid model. Choose one of the models shown above.")


def prepare_dataset(city_name: str) -> None:
    """Download, preprocess, and chronologically split the selected data."""
    latitude, longitude = get_coordinates(city_name)
    raw_output = RAW_DIR / f"{city_name}.csv"

    # Historical services generally have complete hourly data through yesterday.
    end_date = (date.today() - timedelta(days=1)).isoformat()

    print("\nDownloading weather history...\n")
    save_historical_weather(
        latitude=latitude,
        longitude=longitude,
        start_date="2015-01-01",
        end_date=end_date,
        output_file=str(raw_output),
    )

    print("\nPreprocessing...\n")
    preprocess()

    print("\nSplitting datasets...\n")
    split_dataset()

    print("\n==============================")
    print("Dataset Ready")
    print("==============================")


def run_training(model_name: str, module_name: str) -> None:
    """Run training with the same Python interpreter as weather_main.py."""
    print(f"\nStarting {model_name.upper()} training...\n")

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    create_project_directories()

    print("\n==============================")
    print("Weather Temperature AI")
    print("==============================")

    city_name = choose_city()
    prepare_dataset(city_name)

    models = available_models()
    model_choice = choose_model(models)
    run_training(model_choice, models[model_choice])

    print("\nTraining completed successfully.")

    if model_choice == "lstm":
        print("\nGenerate a live LSTM forecast with:")
        print(f'  "{sys.executable}" -m src.models.lstm.predict --city {city_name}')


if __name__ == "__main__":
    main()