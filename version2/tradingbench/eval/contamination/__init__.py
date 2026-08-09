"""Contamination harness: masks, probes, synthetic controls, leak detection."""

from tradingbench.eval.contamination.alias_map import AliasMap, MASKS, DECISION_MODES
from tradingbench.eval.contamination.blind import anonymize_observation, make_blind_map

__all__ = [
    "AliasMap",
    "MASKS",
    "DECISION_MODES",
    "anonymize_observation",
    "make_blind_map",
]
