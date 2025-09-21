# Thin compatibility layer so older imports `from src.feature_lib import build_rich_features`
# continue to work. The actual implementation lives in features_ta.py.
from src.features_ta import build_rich_features

__all__ = ["build_rich_features"]