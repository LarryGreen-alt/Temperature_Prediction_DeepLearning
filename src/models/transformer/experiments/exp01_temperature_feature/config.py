from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp01_temperature_feature"

# Same hyperparameters as the baseline Transformer run, so the only variable
# between this experiment and the baseline is the input features below.
CONFIG = TransformerConfig()

# Baseline features plus the target's own recent history: does giving the
# model its own past temperature readings (in addition to the other weather
# covariates) improve forecasts over weather-only inputs?
FEATURE_COLUMNS = data.FEATURE_COLUMNS + [data.TARGET_COLUMN]
