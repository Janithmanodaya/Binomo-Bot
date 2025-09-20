import argparse
import os
import math
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import ccxt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class CostModel:
    taker_fee_bps: float = 4.0       # per side, in basis points
    slippage_bps: float = 1.0        # per side, in basis points

    @property
    def roundtrip_cost_ret(self) -> float:
        # Convert bps cost per side to total roundtrip return threshold (decimal)
        # roundtrip bps = 2 * (fee + slippage)
        return 2.0 * (self.taker_fee_bps + self.slippage_bps) / 10000.0


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    for d in ["data", "data/raw", "data/processed"]:
        os.makedirs(d, exist_ok=True)


def utc_ms(dt: pd.Timestamp) -> int:
    return int(pd.Timestamp(dt, tz="UTC").value // 1_000_000)


def fetch_ohlcv_ccxt(symbol: str, days: int, exchange: str = "binance") -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV from ccxt with pagination.
    """
    if exchange != "binance":
        raise NotImplementedError("Only binance spot is implemented in this baseline.")
    ex = ccxt.binance({"enableRateLimit": True})
    timeframe = "1m"
    limit = 1000
    since = utc_ms(pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=days))

    all_rows: List[List[float]] = []
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        # Stop if no more new data
        last_ts = ohlcv[-1][0]
        # Advance since by last_ts + 60s
        since = last_ts + 60_000
        # Respect rate limit
        time.sleep(ex.rateLimit / 1000.0)
        # Prevent runaway loops
        if len(ohlcv) < limit:
            break

    if not all_rows:
        raise RuntimeError("No OHLCV data fetched. Check symbol or connectivity.")

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    return df


# ----------------------------
# Feature Engineering
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a compact set of cost-aware minute features.
    All features use only information available at or before t.
    """
    out = df.copy()
    out["ret_1"] = np.log(out["close"]).diff()
    out["ret_3"] = out["ret_1"].rolling(3).sum()
    out["ret_5"] = out["ret_1"].rolling(5).sum()
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_15"] = out["ret_1"].rolling(15).std()

    out["ema_3"] = ema(out["close"], 3)
    out["ema_9"] = ema(out["close"], 9)
    out["ema_21"] = ema(out["close"], 21)
    out["ema_3_delta"] = out["ema_3"] / out["close"] - 1.0
    out["ema_9_delta"] = out["ema_9"] / out["close"] - 1.0
    out["ema_21_delta"] = out["ema_21"] / out["close"] - 1.0

    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14) / out["close"]

    # Candle shape features
    body = (out["close"] - out["open"]).abs()
    range_ = (out["high"] - out["low"]).replace(0, np.nan)
    out["candle_body_frac"] = (body / range_).fillna(0.0)
    out["upper_wick_frac"] = ((out["high"] - out[["open", "close"]].max(axis=1)) / range_).clip(lower=0).fillna(0.0)
    out["lower_wick_frac"] = (((out[["open", "close"]].min(axis=1)) - out["low"]) / range_).clip(lower=0).fillna(0.0)

    # Temporal cyclic encodings
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
    for col in ["ret_1", "ret_3", "ret_5"]:
        mean = out[col].rolling(200).mean()
        std = out[col].rolling(200).std().replace(0, np.nan)
        out[col + "_z"] = ((out[col] - mean) / std).fillna(0.0)

    out = out.dropna().copy()
    return out


# ----------------------------
# Labeling
# ----------------------------
def build_labels(features: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    """
    Create thresholded UP/DOWN labels using next return and a cost-aware tau.
    """
    df = features.copy()
    # Next-minute log return
    df["next_ret"] = np.log(df["close"]).diff().shift(-1)
    tau = cost.roundtrip_cost_ret
    # 1 for UP, 0 for DOWN, NaN for NEUTRAL (within +/- tau)
    df["label"] = np.where(df["next_ret"] > tau, 1, np.where(df["next_ret"] < -tau, 0, np.nan))
    # Drop last bar (no next)
    df = df.iloc[:-1].copy()
    return df


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    exclude = {
        "open", "high", "low", "close", "volume",
        "label", "next_ret",
        "minute", "hour", "dow"
    }
    feats = [c for c in df.columns if c not in exclude]
    X = df[feats]
    y = df["label"]
    return X, y


# ----------------------------
# Walk-forward training and evaluation
# ----------------------------
def rolling_windows(df: pd.DataFrame, folds: int, val_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Produce expanding train windows and fixed-length validation windows (by calendar days).
    Returns list of tuples: (train_start, train_end, val_start, val_end)
    """
    start = df.index.min()
    end = df.index.max()
    total_days = (end - start).days
    if total_days < (val_days * folds + 5):
        raise ValueError("Not enough data for the requested folds/val_days.")

    windows = []
    train_start = start
    val_start = df.index.min() + pd.Timedelta(days=total_days - val_days * folds)
    for i in range(folds):
        vs = val_start + pd.Timedelta(days=i * val_days)
        ve = vs + pd.Timedelta(days=val_days)
        te = vs
        windows.append((train_start, te, vs, ve))
    return windows


def tune_threshold_for_pnl(probs: pd.Series, next_ret: pd.Series, cost: CostModel) -> float:
    """
    Sweep decision thresholds and pick the one that maximizes average PnL after costs.
    Strategy:
      - Long if p > t
      - Short if p < 1 - t
      - Else flat
    """
    thresholds = np.linspace(0.5, 0.7, 21)  # focus on confident trades
    best_t = 0.55
    best_pnl = -1e9
    rt_cost = cost.roundtrip_cost_ret

    for t in thresholds:
        long_mask = probs > t
        short_mask = probs < (1 - t)
        pnl = np.where(long_mask, next_ret - rt_cost, 0.0) + np.where(short_mask, -next_ret - rt_cost, 0.0)
        avg = pnl.mean()
        if avg > best_pnl:
            best_pnl = avg
            best_t = t
    return float(best_t)


def simulate_pnl(probs: pd.Series, next_ret: pd.Series, threshold: float, cost: CostModel) -> pd.DataFrame:
    """
    Return per-minute signals and PnL.
    """
    rt_cost = cost.roundtrip_cost_ret
    signal = np.where(probs > threshold, 1, np.where(probs < 1 - threshold, -1, 0))
    pnl = np.where(signal == 1, next_ret - rt_cost, np.where(signal == -1, -next_ret - rt_cost, 0.0))
    out = pd.DataFrame(
        {
            "prob_up": probs,
            "signal": signal,
            "next_ret": next_ret,
            "pnl": pnl,
        },
        index=probs.index,
    )
    return out


def sharpe_ratio(returns: pd.Series, minutes_per_year: int = 525600) -> float:
    if returns.std() == 0:
        return 0.0
    # Scale per-minute returns to annualized Sharpe
    return float((returns.mean() / returns.std()) * math.sqrt(minutes_per_year))


def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = (cum - roll_max)
    return float(dd.min())


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.Booster:
    # Drop NaNs in labels
    tr_mask = y_train.notna()
    vl_mask = y_val.notna()
    dtrain = lgb.Dataset(X_train[tr_mask], label=y_train[tr_mask])
    dval = lgb.Dataset(X_val[vl_mask], label=y_val[vl_mask])
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        verbose=-1,
        seed=42,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def run_pipeline(
    symbol: str,
    days: int,
    cost: CostModel,
    folds: int,
    val_days: int,
    default_prob_threshold: float,
):
    ensure_dirs()

    # 1) Data
    print(f"Fetching {days} days of 1m OHLCV for {symbol} from Binance...")
    ohlcv = fetch_ohlcv_ccxt(symbol, days)
    raw_path = f"data/raw/{symbol.replace('/', '')}_1m.parquet"
    ohlcv.to_parquet(raw_path)
    print(f"Saved raw OHLCV: {raw_path} ({len(ohlcv):,} rows)")

    # 2) Features
    feats = build_features(ohlcv)

    # 3) Labels
    labeled = build_labels(feats, cost)
    X, y = feature_target_split(labeled)

    # 4) Walk-forward windows
    windows = rolling_windows(labeled, folds=folds, val_days=val_days)
    all_val_frames = []
    summary_rows = []

    for i, (tr_s, tr_e, va_s, va_e) in enumerate(windows, start=1):
        tr_mask = (labeled.index >= tr_s) & (labeled.index < tr_e)
        va_mask = (labeled.index >= va_s) & (labeled.index < va_e)
        X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
        X_va, y_va = X.loc[va_mask], y.loc[va_mask]
        next_ret_va = labeled.loc[va_mask, "next_ret"]

        print(f"\nFold {i}: train {tr_s} to {tr_e} ({len(X_tr):,} rows), validate {va_s} to {va_e} ({len(X_va):,} rows)")

        if y_tr.notna().sum() < 1000 or y_va.notna().sum() < 500:
            print("Insufficient labeled samples for this fold; skipping.")
            continue

        model = train_lightgbm(X_tr, y_tr, X_va, y_va)

        # Predict probabilities for validation
        va_mask_notna = y_va.notna()
        proba = pd.Series(index=X_va.index, dtype=float)
        proba.loc[va_mask_notna] = model.predict(X_va[va_mask_notna], num_iteration=model.best_iteration)

        # Threshold tuning by validation PnL
        if proba.notna().any():
            t_opt = tune_threshold_for_pnl(proba.dropna(), next_ret_va.loc[proba.dropna().index], cost)
        else:
            t_opt = default_prob_threshold

        sim = simulate_pnl(proba.fillna(0.5), next_ret_va, t_opt, cost)
        sim["fold"] = i

        # Stats
        y_hat_cls = pd.Series(np.where(sim["prob_up"] >= 0.5, 1, 0), index=sim.index)
        mask_valid = y_va.notna()
        auc = roc_auc_score(y_va[mask_valid], sim.loc[mask_valid, "prob_up"]) if mask_valid.sum() > 0 else np.nan
        acc = accuracy_score(y_va[mask_valid], y_hat_cls[mask_valid]) if mask_valid.sum() > 0 else np.nan

        sharpe = sharpe_ratio(sim["pnl"])
        mdd = max_drawdown(sim["pnl"].cumsum())
        pf = profit_factor(sim["pnl"])
        expectancy = sim["pnl"].mean()

        summary_rows.append(
            dict(
                fold=i,
                threshold=t_opt,
                auc=auc,
                accuracy=acc,
                expectancy=expectancy,
                sharpe=sharpe,
                max_drawdown=mdd,
                profit_factor=pf,
                trades=(sim["signal"] != 0).sum(),
                samples=len(sim),
            )
        )

        all_val_frames.append(sim)

    if not all_val_frames:
        raise RuntimeError("No validation results produced. Try increasing days or reducing folds/val_days.")

    results = pd.concat(all_val_frames).sort_index()
    out_csv = "data/processed/predictions.csv"
    results.to_csv(out_csv, index_label="timestamp")
    print(f"\nSaved per-minute predictions and PnL to: {out_csv}")

    summary = pd.DataFrame(summary_rows)
    print("\nSummary (per-fold):")
    with pd.option_context("display.max_columns", None):
        print(summary.round(6))

    print("\nAggregate metrics:")
    print(f"- Total samples: {len(results):,}")
    print(f"- Total trades: {(results['signal'] != 0).sum():,}")
    print(f"- Mean PnL per minute: {results['pnl'].mean():.6e}")
    print(f"- Sharpe: {sharpe_ratio(results['pnl']):.3f}")
    print(f"- Profit factor: {profit_factor(results['pnl']):.3f}")
    print(f"- Max drawdown (cum PnL units): {max_drawdown(results['pnl'].cumsum()):.6f}")


def parse_args():
    p = argparse.ArgumentParser(description="Cost-aware 1-minute crypto direction baseline with LightGBM")
    p.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol on Binance spot, e.g., BTC/USDT")
    p.add_argument("--days", type=int, default=60, help="Lookback days of 1m OHLCV to fetch")
    p.add_argument("--taker-fee-bps", type=float, default=4.0, help="Taker fee per side (bps)")
    p.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage per side (bps)")
    p.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    p.add_argument("--val-days", type=int, default=10, help="Validation days per fold")
    p.add_argument("--prob-threshold", type=float, default=0.55, help="Default decision threshold if tuning fails")
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cost = CostModel(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)
    run_pipeline(
        symbol=args.symbol,
        days=args.days,
        cost=cost,
        folds=args.folds,
        val_days=args.val_days,
        default_prob_threshold=args.prob_threshold,
    )