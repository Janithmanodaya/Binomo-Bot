# Compatibility layer: expose rich TA features and enriched (macro/on-chain) features.
from src.features_ta import build_rich_features  # existing
from src.features_external import build_enriched_features  # new

__all__ = ["build_rich_features", "build_enriched_features"]