import time
from typing import Optional, Callable, List, Tuple

import pandas as pd
import numpy as np
import ccxt
import lightgbm as lgb

from src.run_pipeline import build_features, CostModel, utc_ms


def fetch_recent_ohlcv(symbol: str, minutes: int = 400) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    timeframe = "1m"
    limit = min(minutes + 5, 1000)
    # Fetch recent candles without since to get latest window
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("No OHLCV data returned.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df


def round_up_to_next_minute(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return (ts.floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")


def wait_until(target_ts: pd.Timestamp, sleep_sec: float = 0.5):
    while pd.Timestamp.now(tz="UTC") < target_ts:
        time.sleep(sleep_sec)


def live_loop(
    symbol: str,
    model_path: str,
    feature_names: List[str],
    threshold: float,
    cost: CostModel,
    minutes_to_run: int = 5,
    on_update: Optional[Callable[[dict], None]] = None,
):
    """
    Live monitoring:
      - Fetch latest candles
      - Build features
      - Predict prob_up for the last complete bar -> produce signal for next minute
      - After one new minute closes, fetch again and score correctness
    """
    model = lgb.Booster(model_file=model_path)

    # Align start at next minute boundary
    now = pd.Timestamp.now(tz="UTC")
    next_min = round_up_to_next_minute(now)
    if on_update:
        on_update({"msg": f"Waiting for next minute boundary: {next_min} UTC"})
    wait_until(next_min)

    for i in range(minutes_to_run):
        # Step 1: fetch recent window and build features
        raw = fetch_recent_ohlcv(symbol, minutes=500)
        feats = build_features(raw)
        if feats.empty:
            if on_update:
                on_update({"msg": "Insufficient features; waiting next minute..."})
            wait_until(round_up_to_next_minute(pd.Timestamp.now(tz="UTC")))
            continue

        # Use the last available row as current state (t)
        current_ts = feats.index[-1]
        X_last = feats.iloc[[-1]][feature_names].copy()
        # Fill any missing columns if mismatch (robustness)
        for col in feature_names:
            if col not in X_last.columns:
                X_last[col] = 0.0
        X_last = X_last[feature_names]

        prob_up = float(model.predict(X_last)[0])
        signal = int(1 if prob_up > threshold else (-1 if prob_up < 1 - threshold else 0))

        if on_update:
            on_update({
                "timestamp": str(current_ts),
                "prob_up": prob_up,
                "signal": signal,
                "msg": f"[{i+1}/{minutes_to_run}] Current prob_up={prob_up:.4f}, signal={signal} at {current_ts}",
            })

        # Step 2: wait until next minute closes then check correctness
        next_bar_ts = round_up_to_next_minute(current_ts)
        wait_until(next_bar_ts + pd.Timedelta(seconds=2))  # small buffer

        # Fetch again to include the next bar
        raw2 = fetch_recent_ohlcv(symbol, minutes=500)
        if current_ts not in raw2.index or next_bar_ts not in raw2.index:
            # If exchange lag, skip evaluation but continue
            if on_update:
                on_update({"msg": "Next bar not available yet; skipping evaluation."})
            continue

        # Compute realized next_ret
        c0 = raw2.loc[current_ts, "close"]
        c1 = raw2.loc[next_bar_ts, "close"]
        next_ret = float(np.log(c1) - np.log(c0))

        # Determine correctness: UP correct if signal=1 and next_ret > tau; DOWN correct if signal=-1 and next_ret < -tau; FLAT correct if within [-tau, +tau]
        tau = cost.roundtrip_cost_ret
        correct = None
        if signal == 1:
            correct = next_ret > tau
        elif signal == -1:
            correct = next_ret < -tau
        else:
            correct = abs(next_ret) <= tau

        if on_update:
            on_update({
                "timestamp_eval": str(next_bar_ts),
                "next_ret": next_ret,
                "correct": bool(correct),
                "msg": f"Evaluated at {next_bar_ts}: next_ret={next_ret:.6e}, correct={correct}",
            })