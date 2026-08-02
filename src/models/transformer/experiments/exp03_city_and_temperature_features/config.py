from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp03_city_and_temperature_features"

# Same hyperparameters as the baseline Transformer run, so the only variable
# between this experiment and the baseline is the combination of inputs
# below (exp01's feature + exp02's city embedding, applied together).
CONFIG = TransformerConfig()

# exp01's change: baseline features plus the target's own recent history.
FEATURE_COLUMNS = data.FEATURE_COLUMNS + [data.TARGET_COLUMN]

# exp02's change: city identity injected as a learned embedding (see
# train.py's city_aware=True and architecture.py's build_model()).
CITY_EMBED_DIM = 8
