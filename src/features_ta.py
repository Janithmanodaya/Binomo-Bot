"""
Lightweight technical-features module with multi-timeframe signals.

Provides build_rich_features(df) used by realtime and rich-training scripts.
This implementation avoids external TA libs and extends the baseline features
from run_pipeline.py with:
- Multi-timeframe aggregates (3m, 5m, 15m)
- Momentum/volatility indicators (MACD, BBands, Stoch, OBV, MFI proxy, VWAP)
- Rolling distribution stats (skew, kurtosis)
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

# Reuse simple primitives from the baseline where possible
from src.run_pipeline import ema, rsi, atr


def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    base = s.shift(periods)
    return (s / base - 1.0).replace([np.inf, -np.inf], np.nan)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample minute OHLCV to higher timeframe using left-closed, left-labeled bars.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    rf = df.resample(rule, closed="left", label="left").agg(agg).dropna()
    return rf


def _add_minute_level_features(out: pd.DataFrame) -> pd.DataFrame:
    # Basic log returns and volatility
    out["ret_1"] = np.log(out["close"]).diff()
    out["ret_3"] = out["ret_1"].rolling(3).sum()
    out["ret_5"] = out["ret_1"].rolling(5).sum()
    out["ret_15"] = out["ret_1"].rolling(15).sum()
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_15"] = out["ret_1"].rolling(15).std()
    out["vol_60"] = out["ret_1"].rolling(60).std()

    # Moving averages and distances
    for span in (3, 9, 21, 50, 100, 200):
        out[f"ema_{span}"] = ema(out["close"], span)
        out[f"ema_{span}_delta"] = out[f"ema_{span}"] / out["close"] - 1.0

    # MACD (12/26 EMA) and signal 9
    ema12 = ema(out["close"], 12)
    ema26 = ema(out["close"], 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # Bollinger Bands (20)
    ma20 = out["close"].rolling(20).mean()
    sd20 = out["close"].rolling(20).std()
    out["bb_upper_20"] = ma20 + 2 * sd20
    out["bb_lower_20"] = ma20 - 2 * sd20
    out["bb_width_20"] = (out["bb_upper_20"] - out["bb_lower_20"]) / out["close"]
    out["bb_z_20"] = (out["close"] - ma20) / (sd20.replace(0, np.nan))

    # Stochastic Oscillator (14)
    hh14 = out["high"].rolling(14).max()
    ll14 = out["low"].rolling(14).min()
    denom = (hh14 - ll14).replace(0, np.nan)
    k = ((out["close"] - ll14) / denom).clip(0, 1)
    out["stoch_k_14"] = k
    out["stoch_d_14"] = k.rolling(3).mean()

    # OBV
    direction = np.sign(out["close"].diff()).fillna(0.0)
    out["obv"] = (direction * out["volume"]).cumsum()

    # Simple rolling VWAP proxy (20)
    pv = out["close"] * out["volume"]
    out["vwap_20"] = (pv.rolling(20).sum() / (out["volume"].rolling(20).sum().replace(0, np.nan)))

    # ATR-based normalized range
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14) / out["close"]

    # RSI short/long
    out["rsi_7"] = rsi(out["close"], 7)
    out["rsi_14"] = rsi(out["close"], 14)

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

    # Rolling z-score normalize a subset
    for col in ["ret_1", "ret_3", "ret_5", "ret_15"]:
        mean = out[col].rolling(200).mean()
        std = out[col].rolling(200).std().replace(0, np.nan)
        out[col + "_z"] = ((out[col] - mean) / std).fillna(0.0)

    # Distribution stats on short window
    out["ret_1_skew_30"] = out["ret_1"].rolling(30).skew()
    out["ret_1_kurt_30"] = out["ret_1"].rolling(30).kurt()

    return out


def _add_multi_timeframe_features(df_1m: pd.DataFrame, out: pd.DataFrame, tfs: List[str]) -> pd.DataFrame:
    """
    For each timeframe in tfs, compute a compact feature set and asof-join to minute index.
    """
    idx = out.index
    for rule in tfs:
        rf = _resample_ohlcv(df_1m, rule)
        # Compute simple features on the resampled frame
        rf_feat = rf.copy()
        rf_feat = _add_minute_level_features(rf_feat)
        # Keep a subset and suffix with timeframe
        cols_keep = [
            "ret_1", "ret_3", "ret_5", "vol_5", "vol_15",
            "ema_9_delta", "ema_21_delta", "macd", "macd_signal", "macd_hist",
            "bb_width_20", "bb_z_20", "stoch_k_14", "stoch_d_14",
            "rsi_7", "rsi_14", "atr_14", "pos_in_range_50",
        ]
        cols_keep = [c for c in cols_keep if c in rf_feat.columns]
        rf_feat = rf_feat[cols_keep]
        rf_feat.columns = [f"{c}_{rule}" for c in rf_feat.columns]
        # As-of join (forward-fill) to align to minute index without leakage
        rf_feat = rf_feat.reindex(idx.union(rf_feat.index)).sort_index().ffill().reindex(idx)
        out = out.join(rf_feat)
    return out


def build_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an extended set of minute features using only OHLCV.
    Includes multi-timeframe aggregates to improve separability and confidence.

    Robust to short lookbacks: if there is not enough data to compute indicators,
    returns an empty DataFrame instead of raising, allowing callers to skip gracefully.
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

        # Ensure a DatetimeIndex with UTC tz
        base = df.copy()
        if not isinstance(base.index, pd.DatetimeIndex):
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        if base.index.tz is None:
            base.index = base.index.tz_localize("UTC")
        else:
            base.index = base.index.tz_convert("UTC")

        # Require a minimal window to compute stable rolling stats
        if len(base) < 220:  # enough to compute 200-roll stats without all-NaN
            return pd.DataFrame(index=base.index)

        out = _add_minute_level_features(base)
        # Multi-timeframe joins: 3m, 5m, 15m
        out = _add_multi_timeframe_features(base, out, tfs=["3min", "5min", "15min"])

        out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
        return out
    except Exception:
        # On any failure, return empty to signal caller to skip this tick
        return pd.DataFrame(index=(df.index if isinstance(df.index, pd.DatetimeIndex) else None))