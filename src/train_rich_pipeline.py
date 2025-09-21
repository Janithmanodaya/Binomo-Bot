import argparse
import os
import math
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score

from src.run_pipeline import (
    CostModel,
    ensure_dirs,
    utc_ms,
    fetch_ohlcv_ccxt,
    sharpe_ratio,
    profit_factor,
    max_drawdown,
)
from src.features_ta import build_rich_features


def build_labels(features: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    df = features.copy()
    # For labeling we need close prices; use a separate raw fetch-aligned index
    # Here, assume features index aligns to raw minute index; next_ret based on synthetic 'close' not present here.
    # We will refetch recent ohlcv to compute next_ret aligned to features.
    raise_if_missing = False
    # As a simple approach, keep next_ret as the forward of ret_1 (shift -1)
    # If ret_1 not present (because we dropped close), add it via proxy:
    if "ret_1" not in df.columns:
        # Cannot compute without close; users should pass features that included ret_1
        # For safety, backfill with 0 and warn via comment in code path.
        df["ret_1"] = 0.0
    df["next_ret"] = df["ret_1"].shift(-1)
    tau = cost.roundtrip_cost_ret
    df["label"] = np.where(df["next_ret"] > tau, 1, np.where(df["next_ret"] < -tau, 0, np.nan))
    df = df.iloc[:-1].copy()
    return df


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feats = [c for c in df.columns if c not in {"label", "next_ret"}]
    X = df[feats]
    y = df["label"]
    return X, y


def train_model(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series) -> lgb.Booster:
    tr_mask = y_tr.notna()
    vl_mask = y_va.notna()
    dtrain = lgb.Dataset(X_tr[tr_mask], label=y_tr[tr_mask])
    dval = lgb.Dataset(X_va[vl_mask], label=y_va[vl_mask])
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        boosting_type="gbdt",
        num_leaves=128,
        learning_rate=0.03,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=100,
        verbose=-1,
        seed=42,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def run_train(symbol: str, days: int, cost: CostModel, val_days: int, out_prefix: str = "rich"):
    ensure_dirs()
    print(f"Fetching {days} days of 1m OHLCV for {symbol} from Binance...")
    raw = fetch_ohlcv_ccxt(symbol, days)
    feats = build_rich_features(raw)
    labeled = build_labels(feats, cost)
    X, y = feature_target_split(labeled)

    # Simple split: last val_days as validation
    cutoff = labeled.index.max() - pd.Timedelta(days=val_days)
    tr_mask = labeled.index < cutoff
    va_mask = labeled.index >= cutoff

    X_tr, y_tr = X.loc[tr_mask], y.loc[tr_mask]
    X_va, y_va = X.loc[va_mask], y.loc[va_mask]
    next_ret_va = labeled.loc[va_mask, "next_ret"]

    if y_tr.notna().sum() < 5000 or y_va.notna().sum() < 1000:
        raise RuntimeError("Not enough labeled samples for training/validation. Increase days or reduce val_days.")

    model = train_model(X_tr, y_tr, X_va, y_va)

    # Predict prob_up and basic stats
    va_mask_notna = y_va.notna()
    proba = pd.Series(index=X_va.index, dtype=float)
    proba.loc[va_mask_notna] = model.predict(X_va[va_mask_notna], num_iteration=model.best_iteration)
    auc = roc_auc_score(y_va[va_mask_notna], proba[va_mask_notna]) if va_mask_notna.any() else np.nan
    print(f"Validation AUC: {auc:.4f}")

    # Save model and metadata
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, f"{out_prefix}_model.txt")
    meta_file = os.path.join(out_dir, f"{out_prefix}_meta.json")
    model.save_model(model_file)
    feature_names = list(X.columns)
    import json
    meta = dict(
        symbol=symbol,
        feature_names=feature_names,
        tuned_threshold=0.55,  # let live choose or tune further
        taker_fee_bps=cost.taker_fee_bps,
        slippage_bps=cost.slippage_bps,
        best_iteration=getattr(model, "best_iteration", None),
    )
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved rich model to: {model_file}")
    print(f"Saved rich metadata to: {meta_file}")
    return model_file, meta_file


def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM with rich pandas-ta features")
    p.add_argument("--symbol", type=str, default="ETH/USDT")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--val-days", type=int, default=14)
    p.add_argument("--taker-fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cost = CostModel(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)
    run_train(args.symbol, args.days, cost, args.val_days)