import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb

from src.features_ta import build_rich_features
from src.run_pipeline import CostModel


def fetch_recent(symbol: str, limit: int = 1000) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=min(limit, 1000))
    if not rows:
        raise RuntimeError("No OHLCV data fetched.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df


def next_minute_boundary(ts: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if ts is None else pd.Timestamp(ts).tz_convert("UTC")
    return (t.floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")


def wait_until(ts: pd.Timestamp, sleep=0.5):
    while pd.Timestamp.now(tz="UTC") < ts:
        time.sleep(sleep)


def confidence_from_prob(prob_up: float, threshold: float) -> float:
    """
    Decision-aware confidence in [0,1]:
      - If long (p >= t): (p - t) / (1 - t)
      - If short (p <= 1 - t): ((1 - t) - p) / (1 - t)
      - Else: 0 (flat)
    """
    lo = 1.0 - threshold
    if prob_up >= threshold:
        return max(0.0, min(1.0, (prob_up - threshold) / (1.0 - threshold)))
    if prob_up <= lo:
        return max(0.0, min(1.0, (lo - prob_up) / (1.0 - threshold)))
    return 0.0


def continuous_predict(
    symbol: str,
    model_path: str,
    feature_names: List[str],
    threshold: float,
    cost: CostModel,
    on_tick: Optional[Callable[[Dict], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    min_confidence: float = 0.0,
    report_path: Optional[str] = None,
):
    """
    Continuously predict every minute:
      - Fetch recent 1m OHLCV
      - Build rich features (including multi-timeframe and many indicators)
      - Predict prob_up using model
      - Emit signal and confidence
      - Wait for next minute close, evaluate correctness, repeat

    Additionally:
      - If report_path is provided, append/overwrite a CSV with per-minute rows
      - Use min_confidence to mark which signals are 'eligible' for metrics

    stop_flag: callable returning True to terminate loop
    """
    model = lgb.Booster(model_file=model_path)

    # Ensure report directory exists
    rows: List[Dict] = []
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Align to next minute
    nb = next_minute_boundary()
    if on_tick:
        on_tick({"msg": f"Waiting for next minute boundary: {nb} UTC"})
    wait_until(nb)

    while True:
        if stop_flag and stop_flag():
            if on_tick:
                on_tick({"msg": "Stop requested; exiting continuous predictor."})
            break

        # Fetch and build features
        raw = fetch_recent(symbol, limit=800)
        feats = build_rich_features(raw)
        if feats.empty:
            if on_tick:
                on_tick({"msg": "No features available; waiting next minute."})
            wait_until(next_minute_boundary())
            continue

        # Current timestamp is last complete candle
        current_ts = feats.index[-1]
        X_last = feats.iloc[[-1]].copy()

        # Align columns to training feature_names
        for col in feature_names:
            if col not in X_last.columns:
                X_last[col] = 0.0
        X_last = X_last[feature_names]

        prob_up = float(model.predict(X_last)[0])
        signal = int(1 if prob_up > threshold else (-1 if prob_up < 1 - threshold else 0))
        conf = confidence_from_prob(prob_up, threshold)

        # Pre-store row
        row: Dict[str, object] = {
            "timestamp": current_ts,
            "prob_up": prob_up,
            "confidence": conf,
            "signal": signal,
        }
        rows.append(row)

        if on_tick:
            on_tick({
                "timestamp": str(current_ts),
                "prob_up": prob_up,
                "confidence": conf,
                "signal": signal,
                "msg": f"Signal at {current_ts}: prob_up={prob_up:.4f}, confidence={conf:.3f}, signal={signal}",
            })

        # Evaluate after next minute close
        next_ts = next_minute_boundary(current_ts)
        wait_until(next_ts + pd.Timedelta(seconds=2))

        raw2 = fetch_recent(symbol, limit=800)
        if current_ts not in raw2.index or next_ts not in raw2.index:
            if on_tick:
                on_tick({"msg": "Next candle not yet available; skipping evaluation."})
            continue
        c0 = float(raw2.loc[current_ts, "close"])
        c1 = float(raw2.loc[next_ts, "close"])
        next_ret = float(np.log(c1) - np.log(c0))
        tau = cost.roundtrip_cost_ret

        if signal == 1:
            correct = next_ret > tau
            pnl = float(next_ret - tau)
        elif signal == -1:
            correct = next_ret < -tau
            pnl = float(-next_ret - tau)
        else:
            correct = abs(next_ret) <= tau
            pnl = 0.0

        # Update last appended row with evaluation
        rows[-1]["eval_timestamp"] = next_ts
        rows[-1]["next_ret"] = next_ret
        rows[-1]["correct"] = bool(correct)
        eligible = (signal != 0) and (conf >= float(min_confidence))
        rows[-1]["eligible"] = bool(eligible)
        rows[-1]["pnl"] = pnl

        # Compute running metrics for eligible signals
        df = pd.DataFrame(rows)
        if "eligible" in df.columns and df["eligible"].any():
            elig = df[df["eligible"] == True]
            trades = int(len(elig))
            wins = int(elig["correct"].sum()) if "correct" in elig.columns else 0
            win_rate = float(wins / trades) if trades > 0 else 0.0
            cum_pnl = float(elig["pnl"].sum()) if "pnl" in elig.columns else 0.0
        else:
            trades = 0
            win_rate = 0.0
            cum_pnl = 0.0

        if report_path:
            # Persist full report on each iteration (small file)
            out_df = df.copy()
            out_df.to_csv(report_path, index=False)

        if on_tick:
            on_tick({
                "timestamp_eval": str(next_ts),
                "next_ret": next_ret,
                "correct": bool(correct),
                "eligible": bool(eligible),
                "metrics": {
                    "eligible_trades": trades,
                    "win_rate": win_rate,
                    "cum_pnl": cum_pnl,
                    "min_confidence": float(min_confidence),
                },
                "last_row": rows[-1].copy(),
                "msg": f"Evaluation at {next_ts}: next_ret={next_ret:.6e}, correct={correct}, eligible={eligible}, trades={trades}, win_rate={win_rate:.3f}, cum_pnl={cum_pnl:.3e}",
            })