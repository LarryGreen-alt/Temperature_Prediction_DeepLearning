from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp01_temperature"

# Same hyperparameters as exp00, so the only variable between this
# experiment and the baseline is the input features below.
CONFIG = TransformerConfig()

# The canonical 11-column list, which already leads with temperature_2m:
# baseline weather covariates plus the target's own recent history. Does
# giving the model its own past temperature readings improve forecasts over
# weather-only inputs, and does the effect decay with forecast horizon?
FEATURE_COLUMNS = data.FEATURE_COLUMNS
