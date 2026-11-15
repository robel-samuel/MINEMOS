"""Natural human language robustness audit for MINEMOS.

The package is deliberately independent of the engine's internals: the
dataset is generated from explicit templates and rule-based noise, and the
harness drives the public ``BlackHoleMemory`` API only.
"""

from .dataset import (
    DEFAULT_SEED,
    DATASET_VERSION,
    generate_cases,
    generate_lifecycle,
    load_jsonl,
    validate_cases,
    validate_lifecycle,
)

__all__ = [
    "DEFAULT_SEED",
    "DATASET_VERSION",
    "generate_cases",
    "generate_lifecycle",
    "load_jsonl",
    "validate_cases",
    "validate_lifecycle",
]
