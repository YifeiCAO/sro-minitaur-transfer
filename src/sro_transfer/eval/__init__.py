"""Evaluation layer (Phases 3-4). Torch imported lazily."""
from .scoring import make_floor_scorer  # noqa: F401
from .identification import identify, identification_report  # noqa: F401
from .nll import nll_floor_real_shuffled  # noqa: F401
