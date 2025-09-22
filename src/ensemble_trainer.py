import os
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from src.run_pipeline import CostModel
from src.advanced_trainer import (
    build_multi_level_labels,
    feature_target_split,
    _class_weights,
    _confidence_from_probs,
)
from src.feature_lib import build_enriched_features


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


def _p_up_cond_from_multiclass(proba: np.ndarray, class_to_idx: Dict[int, int]) -> np.ndarray:
    p_down2 = proba[:, class_to_idx[-2]]
    p_down1 = proba[:, class_to_idx[-1]]
    p_up1 = proba[:, class_to_idx[1]]
    p_up2 = proba[:, class_to_idx[2]]
    p_up_raw = p_up1 + p_up2
    p_down_raw = p_down1 + p_down2
    p_nonflat = p_up_raw + p_down_raw
    with np.errstate(divide="ignore", invalid="ignore"):
        p_up_cond = np.where(p_nonflat > 1e-12, p_up_raw / p_nonflat, 0.5)
    return p_up_cond


def _sweep_threshold_for_pnl(p_up_cond: pd.Series, next_ret: pd.Series, cost: CostModel) -> Tuple[float, float]:
    thresholds = np.linspace(0.58, 0.82, 25)
    rt_cost = cost.roundtrip_cost_ret
    best_t, best_mean = 0.65, -1e9
    for t in thresholds:
        sig = np.where(p_up_cond >= t, 1, np.where(p_up_cond <= 1 - t, -1, 0))
        pnl = np.where(sig == 1, next_ret.values - rt_cost, np.where(sig == -1, -next_ret.values - rt_cost, 0.0))
        m = float(np.mean(pnl))
        if m > best_mean:
            best_mean, best_t = m, float(t)
    return float(best_t), float(best_mean)


def train_ensemble_stacked(
    symbol: str,
    days: int,
    cost: CostModel,
    progress: Optional[Callable[[str, float], None]] = None,
    backtest_days: int = 0,
) -> Tuple[str, str]:
    """
    Train an ensemble of base learners (LightGBM multiclass, RandomForest, MLP) and
    a LogisticRegression meta-learner on their validation outputs (stacking).
    Save the bundle to data/processed/ensemble_advanced_*.
    """
    def report(stage: str, p: float):
        if progress:
            progress(stage, max(0.0, min(1.0, p)))

    ensure_dirs()
    report("Fetching OHLCV", 0.02)
    df = _fetch_ohlcv_days(symbol, days)
    raw_path = f"data/raw/{symbol.replace('/', '')}_1m_ens.parquet"
    df.to_parquet(raw_path)

    report("Building enriched features", 0.10)
    feats = build_enriched_features(df, symbol)

    report("Labeling (multi-level)", 0.18)
    labeled = build_multi_level_labels(feats, cost)
    X_all, y_all = feature_target_split(labeled)
    mask_all = y_all.notna()
    X_all, y_all = X_all[mask_all], y_all[mask_all].astype(int)
    labeled = labeled.loc[X_all.index]

    if len(X_all) < 5000:
        raise RuntimeError("Not enough samples for ensemble training. Increase days.")

    # Optional backtest window
    if backtest_days and backtest_days > 0:
        bt_cut = labeled.index.max() - pd.Timedelta(days=int(backtest_days))
        back_mask = labeled.index >= bt_cut
    else:
        back_mask = pd.Series(False, index=labeled.index)

    # Train/val split (exclude backtest)
    idx_train_val = labeled.index[~back_mask]
    n_tv = len(idx_train_val)
    split = int(n_tv * 0.85)
    tv_idx_sorted = idx_train_val.sort_values()
    idx_tr = tv_idx_sorted[:split]
    idx_va = tv_idx_sorted[split:]

    X_tr, y_tr = X_all.loc[idx_tr], y_all.loc[idx_tr]
    X_va, y_va = X_all.loc[idx_va], y_all.loc[idx_va]
    next_ret_va = labeled.loc[idx_va, "next_ret"]

    classes = [-2, -1, 0, 1, 2]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = y_tr.map(class_to_idx)
    y_va_idx = y_va.map(class_to_idx)

    # Base 1: LightGBM multiclass with class weights
    cw = _class_weights(y_tr, classes)
    class_weight_list = [cw[c] for c in classes]
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

    report("Training base LightGBM", 0.24)
    model_lgb = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    # Base 2: RandomForest (multiclass)
    report("Training base RandomForest", 0.38)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight=None,  # multiclass weights not directly supported well; rely on y distribution
    )
    rf.fit(X_tr, y_tr)

    # Base 3: MLP (multiclass)
    report("Training base MLP", 0.52)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_tr.fillna(0.0), y_tr)

    # Stacking: meta features on validation
    report("Stacking meta-learner", 0.66)
    proba_lgb_va = model_lgb.predict(X_va, num_iteration=getattr(model_lgb, "best_iteration", None))
    p_up_lgb = _p_up_cond_from_multiclass(proba_lgb_va, class_to_idx)

    proba_rf_va = rf.predict_proba(X_va)
    # scikit classes_ ordering; build map
    rf_cls_to_idx = {int(c): i for i, c in enumerate(rf.classes_)}
    # Ensure all 5 classes present; fill missing with zeros
    rf_proba5 = np.zeros((len(X_va), 5), dtype=float)
    for cls, idx in rf_cls_to_idx.items():
        rf_proba5[:, class_to_idx[cls]] = proba_rf_va[:, idx]
    p_up_rf = _p_up_cond_from_multiclass(rf_proba5, class_to_idx)

    proba_mlp_va = mlp.predict_proba(X_va.fillna(0.0))
    mlp_cls_to_idx = {int(c): i for i, c in enumerate(mlp.classes_)}
    mlp_proba5 = np.zeros((len(X_va), 5), dtype=float)
    for cls, idx in mlp_cls_to_idx.items():
        mlp_proba5[:, class_to_idx[cls]] = proba_mlp_va[:, idx]
    p_up_mlp = _p_up_cond_from_multiclass(mlp_proba5, class_to_idx)

    # Meta target: binary up/down only on non-flat minutes
    y_va_np = y_va.values
    nonflat = y_va_np != 0
    y_bin = (y_va_np[nonflat] > 0).astype(int)

    Z_va = np.vstack([p_up_lgb[nonflat], p_up_rf[nonflat], p_up_mlp[nonflat]]).T
    meta = LogisticRegression(max_iter=500, solver="lbfgs")
    meta.fit(Z_va, y_bin)

    # Choose decision threshold on meta outputs by mean PnL
    p_meta_full = meta.predict_proba(np.vstack([p_up_lgb, p_up_rf, p_up_mlp]).T)[:, 1]
    p_meta_s = pd.Series(p_meta_full, index=X_va.index)
    thr, mean_pnl = _sweep_threshold_for_pnl(p_meta_s, next_ret_va, cost)

    # Optional backtest on held-out
    backtest_metrics: Dict[str, float] = {}
    if back_mask.any():
        report("Backtesting (holdout)", 0.78)
        idx_bt = labeled.index[back_mask]
        X_bt = X_all.loc[idx_bt]
        next_ret_bt = labeled.loc[idx_bt, "next_ret"]

        proba_lgb_bt = model_lgb.predict(X_bt, num_iteration=getattr(model_lgb, "best_iteration", None))
        p_up_lgb_bt = _p_up_cond_from_multiclass(proba_lgb_bt, class_to_idx)

        rf_bt = rf.predict_proba(X_bt)
        rf_proba5_bt = np.zeros((len(X_bt), 5), dtype=float)
        for cls, idx in rf_cls_to_idx.items():
            rf_proba5_bt[:, class_to_idx[cls]] = rf_bt[:, idx]
        p_up_rf_bt = _p_up_cond_from_multiclass(rf_proba5_bt, class_to_idx)

        mlp_bt = mlp.predict_proba(X_bt.fillna(0.0))
        mlp_proba5_bt = np.zeros((len(X_bt), 5), dtype=float)
        for cls, idx in mlp_cls_to_idx.items():
            mlp_proba5_bt[:, class_to_idx[cls]] = mlp_bt[:, idx]
        p_up_mlp_bt = _p_up_cond_from_multiclass(mlp_proba5_bt, class_to_idx)

        p_meta_bt = meta.predict_proba(np.vstack([p_up_lgb_bt, p_up_rf_bt, p_up_mlp_bt]).T)[:, 1]
        p_meta_bt_s = pd.Series(p_meta_bt, index=idx_bt)
        t = float(thr)
        rt_cost = cost.roundtrip_cost_ret
        signal_bt = np.where(p_meta_bt_s > t, 1, np.where(p_meta_bt_s < 1 - t, -1, 0))
        pnl_bt = np.where(signal_bt == 1, next_ret_bt - rt_cost, np.where(signal_bt == -1, -next_ret_bt - rt_cost, 0.0))
        trades = int((signal_bt != 0).sum())
        wins = int(((signal_bt == 1) & (next_ret_bt > rt_cost)).sum() + ((signal_bt == -1) & (next_ret_bt < -rt_cost)).sum())
        win_rate = float(wins / trades) if trades > 0 else 0.0
        cum_pnl = float(pnl_bt.sum())
        expectancy = float(np.mean(pnl_bt))
        backtest_metrics = dict(
            trades=trades,
            win_rate=win_rate,
            cum_pnl=cum_pnl,
            expectancy=expectancy,
            days=int(backtest_days),
        )

    # Save bundle
    report("Finalizing ensemble model", 0.84)
    from joblib import dump

    model_dir = "data/processed"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ensemble_advanced_bundle.pkl")
    meta_path = os.path.join(model_dir, "ensemble_advanced_meta.json")

    bundle = dict(
        type="stacked_ensemble_v1",
        lgb_model_path=os.path.join(model_dir, "ensemble_lgb.txt"),
        rf_model_path=os.path.join(model_dir, "ensemble_rf.pkl"),
        mlp_model_path=os.path.join(model_dir, "ensemble_mlp.pkl"),
        meta_model_path=os.path.join(model_dir, "ensemble_meta.pkl"),
    )
    # Save base models
    lgb_path = bundle["lgb_model_path"]
    model_lgb.save_model(lgb_path)
    dump(rf, bundle["rf_model_path"])
    dump(mlp, bundle["mlp_model_path"])
    dump(meta, bundle["meta_model_path"])

    feature_names = list(X_all.columns)
    meta_json = dict(
        symbol=symbol,
        feature_names=feature_names,
        threshold=float(thr),
        taker_fee_bps=float(cost.taker_fee_bps),
        slippage_bps=float(cost.slippage_bps),
        classes=[-2, -1, 0, 1, 2],
        class_to_idx=class_to_idx,
        backtest_metrics=backtest_metrics,
        model_paths=bundle,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    report("Done", 1.0)
    return model_path, meta_path


def load_ensemble_bundle() -> Optional[Dict]:
    meta_path = "data/processed/ensemble_advanced_meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    from joblib import load
    lgb_model = lgb.Booster(model_file=meta["model_paths"]["lgb_model_path"])
    rf = load(meta["model_paths"]["rf_model_path"])
    mlp = load(meta["model_paths"]["mlp_model_path"])
    meta_model = load(meta["model_paths"]["meta_model_path"])
    return dict(
        lgb=lgb_model, rf=rf, mlp=mlp, meta_model=meta_model, meta=meta
    )


def predict_latest_ensemble(
    symbol: str,
    feature_minutes: int,
    bundle: Dict,
    threshold_override: Optional[float] = None,
    extra_features: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Timestamp, float, int, float]:
    """
    Predict on latest minute using stacked ensemble.
    Returns (timestamp, prob_up_meta, signal in {-1,0,1}, confidence)
    """
    lgb_model: lgb.Booster = bundle["lgb"]
    rf: RandomForestClassifier = bundle["rf"]
    mlp: MLPClassifier = bundle["mlp"]
    meta_model: LogisticRegression = bundle["meta_model"]
    meta = bundle["meta"]

    feature_names: List[str] = meta["feature_names"]
    thr: float = float(meta["threshold"])
    if threshold_override is not None:
        try:
            thr = float(threshold_override)
        except Exception:
            pass

    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=max(600, min(feature_minutes, 1000)))
    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = raw.set_index("timestamp").sort_index().drop_duplicates()

    feats = build_enriched_features(raw, symbol)
    if feats.empty:
        raise RuntimeError("Insufficient features for prediction.")
    ts = feats.index[-1]
    X_row = feats.iloc[[-1]].copy()

    if extra_features:
        for k, v in extra_features.items():
            if k in feature_names:
                X_row[k] = float(v)

    for col in feature_names:
        if col not in X_row.columns:
            X_row[col] = 0.0
    X_row = X_row[feature_names]

    # Base predictions
    proba_lgb = lgb_model.predict(X_row)[0]
    p_up_lgb, _, _ = _confidence_from_probs(np.array(proba_lgb, dtype=float), 0.5)

    # Prepare fixed class index map from meta["classes"]
    classes_order = [int(c) for c in meta.get("classes", [-2, -1, 0, 1, 2])]
    class_to_idx = {c: i for i, c in enumerate(classes_order)}

    # RF / MLP class probas to p_up_cond
    rf_proba = rf.predict_proba(X_row)
    rf_cls_to_idx = {int(c): i for i, c in enumerate(rf.classes_)}
    proba5_rf = np.zeros((1, 5), dtype=float)
    for cls, idx in rf_cls_to_idx.items():
        if int(cls) in class_to_idx:
            proba5_rf[0, class_to_idx[int(cls)]] = rf_proba[0, idx]
    p_up_rf = _p_up_cond_from_multiclass(proba5_rf, class_to_idx)[0]

    mlp_proba = mlp.predict_proba(X_row.fillna(0.0))
    mlp_cls_to_idx = {int(c): i for i, c in enumerate(mlp.classes_)}
    proba5_mlp = np.zeros((1, 5), dtype=float)
    for cls, idx in mlp_cls_to_idx.items():
        if int(cls) in class_to_idx:
            proba5_mlp[0, class_to_idx[int(cls)]] = mlp_proba[0, idx]
    p_up_mlp = _p_up_cond_from_multiclass(proba5_mlp, class_to_idx)[0]

    Z = np.array([[p_up_lgb, p_up_rf, p_up_mlp]], dtype=float)
    p_meta = float(meta_model.predict_proba(Z)[0, 1])

    # Decision-aware confidence
    lo = 1.0 - thr
    if p_meta >= thr:
        confidence = float(max(0.0, min(1.0, (p_meta - thr) / (1.0 - thr))))
        signal = 1
    elif p_meta <= lo:
        confidence = float(max(0.0, min(1.0, (lo - p_meta) / (1.0 - thr))))
        signal = -1
    else:
        confidence = 0.0
        signal = 0

    return ts, float(p_meta), int(signal), float(confidence)