"""Reproducible baseline benchmark for binary-treatment uplift modelling."""

from .metrics import evaluate_uplift
from .models import available_models, make_model

__all__ = ["available_models", "evaluate_uplift", "make_model"]
