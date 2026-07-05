from dotenv import load_dotenv
import os

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
SPLIT_DATA_DIR = "data/splits"
MODEL_DIR = "models"

WINDOW_SIZE = 24          # previous 24 hours
FORECAST_HORIZON = 3      # predict 3 hours ahead

TRAIN_SPLIT = 0.70
DEV_SPLIT = 0.15
TEST_SPLIT = 0.15

EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001