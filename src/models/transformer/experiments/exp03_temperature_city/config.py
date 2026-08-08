from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp03_temperature_city"

# Same hyperparameters as exp00, so the only variable between this
# experiment and the baseline is the combination of inputs below (exp01's
# feature plus exp02's city embedding, applied together).
CONFIG = TransformerConfig()

# exp01's change: the canonical 11-column list (temperature history included).
FEATURE_COLUMNS = data.FEATURE_COLUMNS

# exp02's change: city identity injected as a learned embedding (see
# train.py's city_aware=True).
CITY_EMBED_DIM = 8
