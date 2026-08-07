from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp00_baseline"

# The control: no temperature history, no city identity, no baseline blend.
# Every other experiment is a change measured against this one.
CONFIG = TransformerConfig()

# Canonical feature list minus the target itself -- weather covariates only.
FEATURE_COLUMNS = [column for column in data.FEATURE_COLUMNS if column != data.TARGET_COLUMN]
