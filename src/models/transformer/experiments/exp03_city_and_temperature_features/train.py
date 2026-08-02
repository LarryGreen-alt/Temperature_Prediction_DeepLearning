from src.models.common.data import PROJECT_ROOT
from src.models.transformer.training import run_training, timestamp_now
from src.models.transformer.experiments.exp03_city_and_temperature_features.config import (
    CONFIG, EXPERIMENT_NAME, FEATURE_COLUMNS, CITY_EMBED_DIM
)

MODEL_ROOT = PROJECT_ROOT / "models" / "Transformer" / EXPERIMENT_NAME
experiment_dir = MODEL_ROOT / "experiments" / timestamp_now()
checkpoint_path = MODEL_ROOT / "checkpoints" / "best.keras"

run_training(
    CONFIG, experiment_dir, checkpoint_path,
    feature_columns=FEATURE_COLUMNS,
    city_aware=True,
    city_embed_dim=CITY_EMBED_DIM
)
