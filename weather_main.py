from __future__ import annotations

import os
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

# Each menu option launches that model package's train.py module.
MODEL_MODULES = {
    "lstm": "src.models.lstm.train",
    "transformer": "src.models.transformer.train",
    "gru": "src.models.gru.train",
}


def create_project_directories() -> None:
    """Create the shared data/output directories used by the pipeline."""
    for directory in [RAW_DIR, PROCESSED_DIR, SPLIT_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def choose_city() -> str:
    """Prompt until the user selects a city supported by the project."""
    print("\nAvailable Cities\n")
    for city in CITY_COORDINATES:
        print(f" - {city.title()}")

    while True:
        city_name = input("\nEnter city: ").strip().lower()
        if city_name in CITY_COORDINATES:
            return city_name
        print("Invalid city. Please choose one of the listed cities.")


def available_model_modules() -> dict[str, str]:
    """Return only model choices that currently contain a train.py file."""
    available: dict[str, str] = {}

    for model_name, module_name in MODEL_MODULES.items():
        train_file = (
            PROJECT_ROOT
            / "src"
            / "models"
            / model_name
            / "train.py"
        )
        if train_file.exists():
            available[model_name] = module_name

    return available


def choose_model(models: dict[str, str]) -> str:
    """Prompt until the user selects an available model package."""
    if not models:
        raise FileNotFoundError(
            "No model training modules were found. Expected files such as "
            "src/models/lstm/train.py."
        )

    print("\nAvailable Models\n")
    for model_name in models:
        print(f" - {model_name}")

    while True:
        model_choice = input("\nChoose model: ").strip().lower()
        if model_choice in models:
            return model_choice
        print("Invalid model. Please choose one of the listed models.")


def prepare_dataset(city_name: str) -> None:
    """Download, preprocess, and chronologically split the selected data."""
    latitude, longitude = get_coordinates(city_name)
    output_file = RAW_DIR / f"{city_name}.csv"

    # Historical APIs commonly provide complete data through the prior day.
    end_date = (date.today() - timedelta(days=1)).isoformat()

    print("\nDownloading weather history...\n")
    save_historical_weather(
        latitude=latitude,
        longitude=longitude,
        start_date="2015-01-01",
        end_date=end_date,
        output_file=str(output_file),
    )

    print("\nPreprocessing...\n")
    preprocess()

    print("\nSplitting datasets...\n")
    split_dataset()

    required_splits = [
        SPLIT_DIR / "train.csv",
        SPLIT_DIR / "dev.csv",
        SPLIT_DIR / "test.csv",
    ]
    missing = [str(path) for path in required_splits if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset preparation finished, but these required split files "
            f"were not created: {missing}"
        )

    print("\n==============================")
    print("Dataset Ready")
    print("==============================")


def train_selected_model(model_choice: str, module_name: str) -> None:
    """Launch the selected model with the current Python environment."""
    print(f"\nStarting {model_choice.upper()} training...\n")
    print(f"Python environment: {sys.executable}\n")

    # sys.executable keeps TensorFlow and all dependencies in the same active
    # virtual environment that was used to launch weather_main.py.
    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"\n{model_choice.upper()} training completed successfully.")


def main() -> None:
    # Keep relative paths used by existing preprocessing modules anchored to
    # the repository root even if weather_main.py is launched elsewhere.
    os.chdir(PROJECT_ROOT)
    create_project_directories()

    print("\n==============================")
    print("Weather Temperature AI")
    print("==============================")

    city_name = choose_city()
    prepare_dataset(city_name)

    models = available_model_modules()
    model_choice = choose_model(models)
    train_selected_model(model_choice, models[model_choice])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        raise SystemExit(130)
    except subprocess.CalledProcessError as error:
        print(
            "\nTraining failed. The selected training module exited with "
            f"status code {error.returncode}."
        )
        raise SystemExit(error.returncode)
    except Exception as error:
        print(f"\nWeather pipeline failed: {error}")
        raise