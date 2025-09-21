"""
Lightweight technical-features module.

Provides build_rich_features(df) used by realtime and rich-training scripts.
This implementation intentionally avoids external TA libraries to keep
requirements minimal, and extends the baseline features from run_pipeline.py.
"""
from typing import List

import numpy as np
import pandas as pd

# Reuse simple primitives from the baseline where possible
from src.run_pipeline import ema, rsi, atr


def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    base = s.shift(periods)
    return (s / base - 1.0).replace([np.inf, -np.inf], np.nan)


def build_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an extended set of minute features using only OHLCV.
    The output retains the original columns and adds feature columns.
    """
    out = df.copy()

    # Basic log returns and volatility
    out["ret_1"] = np.log(out["close"]).diff()
    out["ret_3"] = out["ret_1"].rolling(3).sum()
    out["ret_5"] = out["ret_1"].rolling(5).sum()
    out["ret_15"] = out["ret_1"].rolling(15).sum()
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_15"] = out["ret_1"].rolling(15).std()
    out["vol_60"] = out["ret_1"].rolling(60).std()

    # Moving averages and distances
    for span in (3, 9, 21, 50, 100):
        out[f"ema_{span}"] = ema(out["close"], span)
        out[f"ema_{span}_delta"] = out[f"ema_{span}"] / out["close"] - 1.0

    # RSI and ATR-based features
    out["rsi_14"] = rsi(out["close"], 14)
    out["rsi_7"] = rsi(out["close"], 7)
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14) / out["close"]

    # Price position within rolling high-low channel
    hh = out["high"].rolling(50).max()
    ll = out["low"].rolling(50).min()
    rng = (hh - ll).replace(0, np.nan)
    out["pos_in_range_50"] = ((out["close"] - ll) / rng).clip(0, 1).fillna(0.5)

    # Candle shape features
    body = (out["close"] - out["open"]).abs()
    range_ = (out["high"] - out["low"]).replace(0, np.nan)
    out["candle_body_frac"] = (body / range_).fillna(0.0)
    out["upper_wick_frac"] = ((out["high"] - out[["open", "close"]].max(axis=1)) / range_).clip(lower=0).fillna(0.0)
    out["lower_wick_frac"] = (((out[["open", "close"]].min(axis=1)) - out["low"]) / range_).clip(lower=0).fillna(0.0)

    # Temporal encodings
    out["minute"] = out.index.minute
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60.0)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    # Normalize some returns with rolling z-scores
    for col in ["ret_1", "ret_3", "ret_5", "ret_15"]:
        mean = out[col].rolling(200).mean()
        std = out[col].rolling(200).std().replace(0, np.nan)
        out[col + "_z"] = ((out[col] - mean) / std).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out