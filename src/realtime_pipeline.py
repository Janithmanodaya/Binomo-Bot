import argparse
import os
import time
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb

from src.run_pipeline import CostModel, tune_threshold_for_pnl, simulate_pnl, sharpe_ratio, max_drawdown, profit_factor
from src.feature_lib import build_rich_features  # thin proxy to features_ta


def ensure_dirs():
    for d in ["data", "data/raw", "data/processed"]:
        os.makedirs(d, exist_ok=True)


def fetch_recent_ohlcv(symbol: str, minutes: int = 300) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    timeframe = "1m"
    limit = min(max(minutes, 50), 1000)
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("No OHLCV data returned.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df


def build_labels_for_training(features: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    df = features.copy()
    df["next_ret"] = np.log(df["close"]).diff().shift(-1)
    tau = cost.roundtrip_cost_ret
    df["label"] = np.where(df["next_ret"] > tau, 1, np.where(df["next_ret"] < -tau, 0, np.nan))
    df = df.iloc[:-1].copy()
    return df


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    exclude = {"open", "high", "low", "close", "volume", "label", "next_ret", "minute", "hour", "dow"}
    feats = [c for c in df.columns if c not in exclude]
    return df[feats], df["label"]


def train_model(symbol: str, days: int, cost: CostModel):
    # Fetch lookback via pagination
    print(f"Fetching ~{days} days of 1m OHLCV for {symbol} for training...")
    # Simple backfill with since-based pagination to get longer history
    ex = ccxt.binance({"enableRateLimit": True})
    timeframe = "1m"
    limit = 1000
    since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        since = ohlcv[-1][0] + 60_000
        time.sleep(ex.rateLimit / 1000.0)
        if len(ohlcv) < limit:
            break

    if not all_rows:
        raise RuntimeError("No data fetched for training.")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()

    ensure_dirs()
    raw_path = f"data/raw/{symbol.replace('/', '')}_1m_train.parquet"
    df.to_parquet(raw_path)
    print(f"Saved training OHLCV to {raw_path} with {len(df):,} rows")

    # Build rich features and labels
    feats = build_rich_features(df)
    labeled = build_labels_for_training(feats, cost)
    X, y = feature_target_split(labeled)

    # Hold out the last N days for validation threshold tuning
    val_len = min(10 * 1440, int(0.2 * len(X)))
    X_tr, y_tr = X.iloc[:-val_len], y.iloc[:-val_len]
    X_va, y_va = X.iloc[-val_len:], y.iloc[-val_len:]
    next_ret_va = labeled.iloc[-val_len:]["next_ret"]

    # Train
    import lightgbm as lgb
    tr_mask, va_mask = y_tr.notna(), y_va.notna()
    dtrain = lgb.Dataset(X_tr[tr_mask], label=y_tr[tr_mask])
    dval = lgb.Dataset(X_va[va_mask], label=y_va[va_mask])
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        boosting_type="gbdt",
        num_leaves=96,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=100,
        verbose=-1,
        seed=42,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    # Tune threshold on validation pnl
    proba_va = pd.Series(index=X_va.index, dtype=float)
    proba_va.loc[va_mask] = model.predict(X_va[va_mask], num_iteration=model.best_iteration)
    threshold = tune_threshold_for_pnl(proba_va.dropna(), next_ret_va.loc[proba_va.dropna().index], cost)

    # Save model and meta
    model_path = "data/processed/realtime_model.txt"
    meta_path = "data/processed/realtime_meta.json"
    model.save_model(model_path)
    meta = dict(
        symbol=symbol,
        feature_names=list(X.columns),
        threshold=float(threshold),
        taker_fee_bps=cost.taker_fee_bps,
        slippage_bps=cost.slippage_bps,
        best_iteration=getattr(model, "best_iteration", None),
    )
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved realtime model to {model_path}")
    print(f"Saved realtime metadata to {meta_path}")
    return model_path, meta_path


def predict_stream(symbol: str, model_path: str, meta_path: str, minutes_to_run: Optional[int] = None, log_to: Optional[str] = None):
    # Load model and meta
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names: List[str] = meta["feature_names"]
    threshold: float = float(meta["threshold"])
    cost = CostModel(taker_fee_bps=float(meta["taker_fee_bps"]), slippage_bps=float(meta["slippage_bps"]))

    model = lgb.Booster(model_file=model_path)

    ensure_dirs()
    out_csv = "data/processed/realtime_signals.csv"
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=["timestamp", "prob_up", "confidence", "signal", "next_check_ts", "correct", "next_ret"]).to_csv(out_csv, index=False)

    def append_row(row: Dict):
        df = pd.DataFrame([row])
        df.to_csv(out_csv, mode="a", index=False, header=False)

    print(f"Starting realtime prediction stream for {symbol}. Output: {out_csv}")
    count = 0
    while True:
        now = pd.Timestamp.now(tz="UTC")
        # Fetch recent window and build features
        raw = fetch_recent_ohlcv(symbol, minutes=600)
        feats = build_rich_features(raw)
        if feats.empty:
            print("Insufficient features; waiting...")
            time.sleep(5)
            continue

        current_ts = feats.index[-1]
        X_last = feats.iloc[[-1]][feature_names].copy()
        # Ensure feature alignment
        for col in feature_names:
            if col not in X_last.columns:
                X_last[col] = 0.0
        X_last = X_last[feature_names]

        prob_up = float(model.predict(X_last)[0])
        confidence = float(max(prob_up, 1 - prob_up))
        signal = int(1 if prob_up > threshold else (-1 if prob_up < 1 - threshold else 0))
        next_ts = (current_ts.floor("T") + pd.Timedelta(minutes=1))

        print(f"{current_ts} | prob_up={prob_up:.4f} conf={confidence:.4f} signal={signal} (threshold={threshold:.2f})")

        # Write immediate row without correctness yet
        append_row(dict(
            timestamp=current_ts.isoformat(),
            prob_up=prob_up,
            confidence=confidence,
            signal=signal,
            next_check_ts=next_ts.isoformat(),
            correct=np.nan,
            next_ret=np.nan,
        ))

        # After next minute closes, check correctness and update (append an eval row)
        # Sleep until few seconds after next minute
        sleep_s = max(2.0, (next_ts - pd.Timestamp.now(tz="UTC")).total_seconds() + 2.0)
        time.sleep(sleep_s)

        raw2 = fetch_recent_ohlcv(symbol, minutes=120)
        if current_ts not in raw2.index or next_ts not in raw2.index:
            print("Next bar not available yet; skipping evaluation.")
        else:
            c0 = float(raw2.loc[current_ts, "close"])
            c1 = float(raw2.loc[next_ts, "close"])
            next_ret = float(np.log(c1) - np.log(c0))
            tau = cost.roundtrip_cost_ret
            if signal == 1:
                correct = next_ret > tau
            elif signal == -1:
                correct = next_ret < -tau
            else:
                correct = abs(next_ret) <= tau

            print(f"Eval {next_ts}: next_ret={next_ret:.6e} correct={bool(correct)}")
            append_row(dict(
                timestamp=next_ts.isoformat(),
                prob_up=prob_up,
                confidence=confidence,
                signal=signal,
                next_check_ts="",
                correct=bool(correct),
                next_ret=next_ret,
            ))

        count += 1
        if minutes_to_run is not None and count >= minutes_to_run:
            print("Realtime prediction stream finished.")
            break


def main():
    p = argparse.ArgumentParser(description="Realtime minute-by-minute prediction with rich features")
    p.add_argument("--symbol", type=str, default="ETH/USDT")
    p.add_argument("--days", type=int, default=90, help="Training lookback")
    p.add_argument("--taker-fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--train", action="store_true", help="Train a realtime model before streaming")
    p.add_argument("--minutes", type=int, default=None, help="How many minutes to run; default infinite")
    args = p.parse_args()

    cost = CostModel(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)
    ensure_dirs()

    model_path = "data/processed/realtime_model.txt"
    meta_path = "data/processed/realtime_meta.json"

    if args.train or not (os.path.exists(model_path) and os.path.exists(meta_path)):
        model_path, meta_path = train_model(args.symbol, args.days, cost)

    predict_stream(args.symbol, model_path, meta_path, minutes_to_run=args.minutes)


if __name__ == "__main__":
    main()