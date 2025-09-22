"""
External (non-OHLCV) features:
- Macro proxies from Yahoo Finance (no API key needed): ^TNX (10Y yield), ^IRX (13W T-bill), GC=F (Gold),
  CL=F (Crude), DXY proxy "DX-Y.NYB" if available.
- Optional on-chain metrics from Glassnode if GLASSNODE_API_KEY is set in env.
  Metrics (daily): active addresses, tx count, transfer volume (native units). Joined and forward-filled.

All external features are resampled/aligned to a minute index by forward-filling.
If a data source is unavailable, the corresponding features are omitted gracefully.
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore


def _download_yf(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        df = yf.download(ticker, start=start.date(), end=end.date() + pd.Timedelta(days=1), progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        return df
    except Exception:
        return None


def load_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch daily macro proxies and align to the provided minute index.
    Produces level and daily pct-change features, forward-filled to minute granularity.
    """
    if len(index) == 0:
        return pd.DataFrame(index=index)

    start = pd.Timestamp(index.min()).tz_convert("UTC") - pd.Timedelta(days=5)
    end = pd.Timestamp(index.max()).tz_convert("UTC") + pd.Timedelta(days=2)

    tickers = {
        "^TNX": "tnx",   # 10Y yield (%)
        "^IRX": "irx",   # 13W T-bill (%)
        "GC=F": "gold",  # Gold futures
        "CL=F": "crude", # Crude futures
        "DX-Y.NYB": "dxy"  # US Dollar Index (sometimes unavailable)
    }

    frames: List[pd.DataFrame] = []
    for tkr, prefix in tickers.items():
        df = _download_yf(tkr, start, end)
        if df is None or df.empty:
            continue
        feat = pd.DataFrame(index=df.index)
        # Use close; for yields, the level is % already
        close = df.get("close", df.iloc[:, 0])
        feat[f"{prefix}_level"] = close.astype(float)
        feat[f"{prefix}_d1"] = close.pct_change().replace([np.inf, -np.inf], np.nan)
        frames.append(feat)

    if not frames:
        return pd.DataFrame(index=index)

    daily = pd.concat(frames, axis=1).dropna(how="all")
    # As-of join to minute index: forward-fill
    macro = daily.reindex(index.union(daily.index)).sort_index().ffill().reindex(index)
    # Use .ffill() instead of deprecated fillna(method="ffill")
    macro = macro.replace([np.inf, -np.inf], np.nan).ffill()
    return macro


def _glassnode_request(asset: str, metric: str, api_key: str, a: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Query a Glassnode metric (daily). Returns DataFrame with datetime index and 'value'.
    """
    url = f"https://api.glassnode.com/v1/metrics/{metric}"
    params: Dict[str, str] = {"api_key": api_key, "a": asset, "i": "24h"}  # daily interval
    if a:
        params["a"] = a
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        if "t" not in df.columns or "v" not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.set_index("timestamp").rename(columns={"v": "value"})[["value"]]
        return df.sort_index()
    except Exception:
        return None


def load_onchain_features(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Optionally fetch daily on-chain metrics from Glassnode for BTC/ETH and align to minute index.
    Requires env GLASSNODE_API_KEY. If absent or fetch fails, returns empty frame.
    """
    api_key = os.getenv("GLASSNODE_API_KEY", "").strip()
    if not api_key or len(index) == 0:
        return pd.DataFrame(index=index)

    base = (symbol.split("/")[0]).upper()
    asset_map = {"BTC": "BTC", "ETH": "ETH"}
    asset = asset_map.get(base, None)
    if asset is None:
        return pd.DataFrame(index=index)

    # Pick a few robust daily series
    # active_addresses, transactions_count, transfer_volume_usd
    metrics = {
        "addresses/active_count": "active_addr",
        "transactions/transfers_count": "tx_count",
        "transactions/transfers_volume_sum": "tx_vol_sum",
    }

    frames = []
    for m, prefix in metrics.items():
        df = _glassnode_request(asset=asset, metric=m, api_key=api_key)
        if df is None or df.empty:
            continue
        feat = pd.DataFrame(index=df.index)
        feat[f"{prefix}_lvl"] = df["value"].astype(float)
        feat[f"{prefix}_d1"] = df["value"].pct_change().replace([np.inf, -np.inf], np.nan)
        frames.append(feat)

    if not frames:
        return pd.DataFrame(index=index)

    daily = pd.concat(frames, axis=1).dropna(how="all")
    oc = daily.reindex(index.union(daily.index)).sort_index().ffill().reindex(index)
    # Use .ffill() instead of deprecated fillna(method="ffill")
    oc = oc.replace([np.inf, -np.inf], np.nan).ffill()
    return oc


def _load_sentiment_from_csv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Optionally load a CSV with columns [timestamp, score] from env SENTIMENT_CSV and align it.
    The resulting column is named 'sentiment_manual' and forward-filled to minute index.
    """
    path = os.getenv("SENTIMENT_CSV", "").strip()
    if not path or not os.path.exists(path) or len(index) == 0:
        return pd.DataFrame(index=index)
    try:
        df = pd.read_csv(path)
        # Try several timestamp field names
        ts_col = None
        for c in ["timestamp", "time", "date", "datetime"]:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            return pd.DataFrame(index=index)
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        # choose score column
        score_col = None
        for c in ["score", "sentiment", "value"]:
            if c in df.columns:
                score_col = c
                break
        if score_col is None:
            return pd.DataFrame(index=index)
        s = df.set_index(ts_col)[score_col].astype(float)
        # sanitize and align
        s = s.replace([np.inf, -np.inf], np.nan).ffill()
        sent = s.reindex(index.union(s.index)).sort_index().ffill().reindex(index)
        return pd.DataFrame({"sentiment_manual": sent}, index=index)
    except Exception:
        return pd.DataFrame(index=index)


def build_enriched_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Build OHLCV technical features and enrich with macro, optional on-chain series,
    and optional sentiment (from CSV if provided via SENTIMENT_CSV env).
    """
    # Local import to avoid circular
    from src.features_ta import build_rich_features

    base = build_rich_features(df)
    if base.empty:
        return base

    macro = load_macro_features(base.index)
    onchain = load_onchain_features(symbol, base.index)
    sent = _load_sentiment_from_csv(base.index)

    frames = [base]
    if not macro.empty:
        frames.append(macro)
    if not onchain.empty:
        frames.append(onchain)
    if not sent.empty:
        frames.append(sent)

    out = pd.concat(frames, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out