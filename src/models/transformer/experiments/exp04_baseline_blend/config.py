from src.models.common import data
from src.models.transformer.training import TransformerConfig

EXPERIMENT_NAME = "exp04_baseline_blend"

# Same inputs as exp03 (temperature history + city embedding); the only new
# variable is baseline_blend, which changes the architecture's head, not the
# inputs. baseline_blend must be True here (not passed separately at
# call time) so it's recorded in this run's config.json.
CONFIG = TransformerConfig(baseline_blend=True)

# Same as exp03: the canonical 11-column list (temperature history included).
FEATURE_COLUMNS = data.FEATURE_COLUMNS

CITY_EMBED_DIM = 8
