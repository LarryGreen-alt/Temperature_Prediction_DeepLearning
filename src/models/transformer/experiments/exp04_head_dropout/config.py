from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp04_head_dropout"

# exp03's config with one value changed: the Dropout in the dense head is
# switched off. The four dropout layers inside the encoder blocks keep
# dropout_rate=0.20, so the only difference between this run and the exp03
# control is the head.
CONFIG = TransformerConfig(head_dropout_rate=0.0)

# Unchanged from exp03: baseline features plus the target's own recent history.
FEATURE_COLUMNS = data.FEATURE_COLUMNS + [data.TARGET_COLUMN]

# Unchanged from exp03: city identity injected as a learned embedding.
CITY_EMBED_DIM = 8
