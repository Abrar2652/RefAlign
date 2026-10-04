"""Data layer: synthetic fault-injected corpus + Who&When loader."""

from .synth import (
    SyntheticConfig,
    generate_corpus,
    AGENTS,
    GOOD_ACTIONS,
    BAD_ACTIONS,
)
from .schema import load_who_and_when

__all__ = [
    "SyntheticConfig",
    "generate_corpus",
    "AGENTS",
    "GOOD_ACTIONS",
    "BAD_ACTIONS",
    "load_who_and_when",
]
