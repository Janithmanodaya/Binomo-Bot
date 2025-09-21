import os
import time
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb

from src.features_ta import build_rich_features
from src.run_pipeline import CostModel, tune_threshold_for_pnl


@dataclass
class AdvancedMeta:
    symbol: str
    feature_names: List[str]
    threshold: float
    taker_fee_bps: float
    slippage_bps: float
    best_iteration: Optional[int]
    classes: List[int]  # [-2,-1,0,1,2]


def ensure_dirs():
    for d in ["data", "data/raw", "data/processed"]:
        os.makedirs(d, exist_ok=True)


def _fetch_ohlcv_days(symbol: str, days: int) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    timeframe = "1m"
    limit = 1000
    since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
    rows: List[List[float]] = []
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        rows.extend(ohlcv)
        since = ohlcv[-1][0] + 60_000
        time.sleep(ex.rateLimit / 1000.0)
        if len(ohlcv) < limit:
            break
    if not rows:
        raise RuntimeError("No OHLCV data fetched.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df


def build_multi_level_labels(features: pd.DataFrame, cost: CostModel) -> pd.DataFrame:
    """
    Multi-level labels using cost-aware thresholds.
    Classes: -2, -1, 0, +1, +2
      -2: next_ret < -2*tau
      -1: -2*tau <= next_ret < -tau
       0: |next_ret| <= tau
      +1: tau < next_ret <= 2*tau
      +2: next_ret > 2*tau
    """
    df = features.copy()
    df["next_ret"] = np.log(df["close"]).diff().shift(-1)
    tau = float(cost.roundtrip_cost_ret)
    bins = [-np.inf, -2 * tau, -tau, tau, 2 * tau, np.inf]
    labels = [-2, -1, 0, 1, 2]
    df["label"] = pd.cut(df["next_ret"], bins=bins, labels=labels, include_lowest=True).astype("float")
    df = df.iloc[:-1].copy()
    return df


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    exclude = {"open", "high", "low", "close", "volume", "label", "next_ret", "minute", "hour", "dow"}
    feats = [c for c in df.columns if c not in exclude]
    X = df[feats]
    y = df["label"]
    return X, y


def _class_weights(y: pd.Series, classes: List[int]) -> Dict[int, float]:
    counts = {c: float((y == c).sum()) for c in classes}
    total = sum(counts.values())
    weights = {}
    for c in classes:
        cnt = counts.get(c, 0.0)
        weights[c] = (total / max(cnt, 1.0)) if cnt > 0 else 1.0
    # normalize weights roughly
    mean_w = np.mean(list(weights.values()))
    return {k: v / mean_w for k, v in weights.items()}


def _confidence_from_probs(prob_vec: np.ndarray, threshold: float) -> Tuple[float, int, float]:
    """
    Map class probability vector [p(-2),p(-1),p(0),p(+1),p(+2)] to:
      - prob_up_cond: conditional P(up | non-flat) = (p(+1)+p(+2)) / (1 - p(0))
      - signal in {-2,-1,0,1,2} using threshold on conditional up/down
      - confidence in [0,1] proportional to margin beyond threshold
    This avoids the model defaulting to 'flat' dominating probability by
    normalizing out the flat mass when making a decision.
    """
    p_down2, p_down1, p_flat, p_up1, p_up2 = prob_vec.tolist()
    p_up_raw = p_up1 + p_up2
    p_down_raw = p_down1 + p_down2
    p_nonflat = p_up_raw + p_down_raw

    if p_nonflat <= 1e-9:
        # Degenerate distribution; return neutral
        return 0.5, 0, 0.0

    # Conditional probabilities given a non-flat move
    p_up = p_up_raw / p_nonflat
    p_down = p_down_raw / p_nonflat

    if p_up >= threshold and p_up >= p_down:
        # strength by which of (+2 or +1) dominates
        strength = 2 if p_up2 >= p_up1 else 1
        conf = (p_up - threshold) / max(1e-9, 1.0 - threshold)
        return float(p_up), strength, max(0.0, min(1.0, float(conf)))
    if p_down >= threshold and p_down > p_up:
        strength = -2 if p_down2 >= p_down1 else -1
        conf = (p_down - threshold) / max(1e-9, 1.0 - threshold)
        # Return p_up (conditional) for a symmetric display; conf relates to the chosen side
        return float(p_up), strength, max(0.0, min(1.0, float(conf)))
    return float(p_up), 0, 0.0


def train_multilevel_model(
    symbol: str,
    days: int,
    cost: CostModel,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, str]:
    """
    Train a multi-level (5-class) LightGBM model and save to data/processed/.
    Returns (model_path, meta_path).
    """
    def report(stage: str, p: float):
        if progress:
            progress(stage, max(0.0, min(1.0, p)))

    ensure_dirs()
    report("Fetching OHLCV", 0.02)
    df = _fetch_ohlcv_days(symbol, days)
    raw_path = f"data/raw/{symbol.replace('/', '')}_1m_adv.parquet"
    df.to_parquet(raw_path)
    report("Building features", 0.10)
    feats = build_rich_features(df)

    report("Labeling (multi-level)", 0.18)
    labeled = build_multi_level_labels(feats, cost)
    X, y = feature_target_split(labeled)

    # Remove NaN labels
    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)

    if len(X) < 5000:
        raise RuntimeError("Not enough samples for advanced training. Increase days.")

    # Train/validation split: last 15% for threshold tuning
    split = int(len(X) * 0.85)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]
    next_ret_va = labeled.iloc[split:]["next_ret"]

    classes = [-2, -1, 0, 1, 2]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = y_tr.map(class_to_idx)
    y_va_idx = y_va.map(class_to_idx)

    # Class weights to handle imbalance
    cw = _class_weights(y_tr, classes)
    # Convert to LightGBM format: list weights by class index order
    class_weight_list = [cw[c] for c in classes]

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr_idx)
    dval = lgb.Dataset(X_va, label=y_va_idx)

    params = dict(
        objective="multiclass",
        num_class=5,
        metric=["multi_logloss"],
        boosting_type="gbdt",
        num_leaves=128,
        learning_rate=0.03,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=200,
        verbose=-1,
        seed=42,
        class_weight=class_weight_list,
    )
    report("Training model", 0.22)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=250, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    report("Tuning threshold", 0.80)

    # Build p_up series for validation to tune threshold
    proba_va = model.predict(X_va, num_iteration=model.best_iteration)  # shape [N,5]
    proba_up = proba_va[:, class_to_idx[1]] + proba_va[:, class_to_idx[2]]
    proba_up_s = pd.Series(proba_up, index=X_va.index)
    threshold = tune_threshold_for_pnl(proba_up_s, next_ret_va.loc[proba_up_s.index], cost)

    report("Saving model", 0.92)
    model_path = "data/processed/advanced_model.txt"
    meta_path = "data/processed/advanced_meta.json"
    model.save_model(model_path)
    meta = AdvancedMeta(
        symbol=symbol,
        feature_names=list(X.columns),
        threshold=float(threshold),
        taker_fee_bps=float(cost.taker_fee_bps),
        slippage_bps=float(cost.slippage_bps),
        best_iteration=getattr(model, "best_iteration", None),
        classes=classes,
    ).__dict__
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report("Done", 1.0)
    return model_path, meta_path


def load_advanced_bundle() -> Optional[Dict]:
    model_path = "data/processed/advanced_model.txt"
    meta_path = "data/processed/advanced_meta.json"
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    booster = lgb.Booster(model_file=model_path)
    return {"booster": booster, "meta": meta}


def predict_latest(
    symbol: str,
    feature_minutes: int,
    bundle: Dict,
) -> Tuple[pd.Timestamp, float, int, float]:
    """
    Predict on the latest completed minute using the advanced model.
    Returns (timestamp, prob_up, signal_strength in {-2..2}, confidence)
    """
    booster: lgb.Booster = bundle["booster"]
    meta = bundle["meta"]
    feature_names: List[str] = meta["feature_names"]
    threshold: float = float(meta["threshold"])

    # Get a recent window straight from the exchange for freshness
    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=max(600, min(feature_minutes, 1000)))
    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = raw.set_index("timestamp").sort_index().drop_duplicates()

    feats = build_rich_features(raw)
    if feats.empty:
        raise RuntimeError("Insufficient features for prediction.")
    ts = feats.index[-1]
    X_row = feats.iloc[[-1]].copy()
    # Align columns
    for col in feature_names:
        if col not in X_row.columns:
            X_row[col] = 0.0
    X_row = X_row[feature_names]

    prob_vec = booster.predict(X_row)[0]  # length 5
    prob_up, signal, confidence = _confidence_from_probs(np.array(prob_vec, dtype=float), threshold)
    return ts, float(prob_up), int(signal), float(confidence)