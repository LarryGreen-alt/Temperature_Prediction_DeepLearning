from src.models.common.data import PROJECT_ROOT
from src.models.transformer.training import TransformerConfig, run_training, timestamp_now

MODEL_NAME = "Transformer"
MODEL_ROOT = PROJECT_ROOT / "models" / MODEL_NAME / "baseline"

config = TransformerConfig()
experiment_dir = MODEL_ROOT / "experiments" / timestamp_now()
checkpoint_path = MODEL_ROOT / "checkpoints" / "best.keras"

run_training(config, experiment_dir, checkpoint_path)
