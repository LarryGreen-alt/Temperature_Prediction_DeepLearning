from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp02_city"

# Same hyperparameters as exp00, so the only variable between this
# experiment and the baseline is whether the model knows which city it's
# predicting for.
CONFIG = TransformerConfig()

# Same weather covariates as exp00 (no temperature history) -- city identity
# is injected separately as a learned embedding, not as a numeric feature
# column (see training.py's city_aware branch and architecture.py's
# build_model()).
FEATURE_COLUMNS = [column for column in data.FEATURE_COLUMNS if column != data.TARGET_COLUMN]

CITY_EMBED_DIM = 8
