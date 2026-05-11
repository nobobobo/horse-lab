"""MLOps helpers for model registry and paper-trading gates."""

from horse_lab.mlops.registry import (
    build_phase4_model_registry_from_report,
    model_registry_to_dict,
)

__all__ = [
    "build_phase4_model_registry_from_report",
    "model_registry_to_dict",
]
