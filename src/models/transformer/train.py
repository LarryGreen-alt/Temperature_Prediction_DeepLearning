"""Thin entrypoint so `python -m src.models.transformer.train` also works,
matching the documented README command. All real logic lives in model.py."""
import src.models.transformer.model  # noqa: F401 (running model.py executes training)
