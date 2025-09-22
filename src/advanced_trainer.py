import os
import time
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
import lightgbm as lgb

from src.feature_lib import build_enriched_features  # enriched TA + macro + (optional) on-chain
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


def _sample_params(rng: np.random.RandomState, class_weight_list: List[float]) -> Dict:
    """Randomly sample a reasonable LightGBM parameter set."""
    return dict(
        objective="multiclass",
        num_class=5,
        metric=["multi_logloss"],
        boosting_type="gbdt",
        num_leaves=int(rng.randint(64, 256)),
        learning_rate=float(10 ** rng.uniform(-2.0, -1.2)),  # ~[0.01, 0.063]
        feature_fraction=float(rng.uniform(0.6, 0.95)),
        bagging_fraction=float(rng.uniform(0.6, 0.95)),
        bagging_freq=int(rng.randint(1, 6)),
        min_data_in_leaf=int(rng.randint(50, 400)),
        max_depth=int(rng.choice([-1, 8, 10, 12])),
        lambda_l1=float(10 ** rng.uniform(-3.0, 1.0)),  # [0.001, 10]
        lambda_l2=float(10 ** rng.uniform(-3.0, 1.0)),
        min_gain_to_split=float(10 ** rng.uniform(-3.0, 0.7)),  # [0.001, 5]
        verbose=-1,
        seed=int(rng.randint(1, 10_000)),
        class_weight=class_weight_list,
        subsample=float(rng.uniform(0.6, 0.95)),  # alias for bagging_fraction in some builds
        feature_pre_filter=False,  # allow dynamic min_data_in_leaf during HPO
    )


def _eval_model_pnl(
    model: "lgb.Booster",
    X_va: pd.DataFrame,
    next_ret_va: pd.Series,
    class_to_idx: Dict[int, int],
    cost: CostModel,
    min_confidence: float = 0.20,
) -> Tuple[float, float]:
    """
    Return (best_threshold, mean_pnl) on validation using cost-aware pnl.

    Uses CONDITIONAL p_up = P(up | non-flat) to match live logic and applies a
    minimum confidence filter when generating trades during threshold search.
    """
    import lightgbm as lgb  # type: ignore

    proba_va = model.predict(X_va, num_iteration=getattr(model, "best_iteration", None))
    if proba_va is None or len(proba_va) == 0:
        return 0.55, -1e9

    # Extract class probabilities
    p_down2 = proba_va[:, class_to_idx[-2]]
    p_down1 = proba_va[:, class_to_idx[-1]]
    p_up1 = proba_va[:, class_to_idx[1]]
    p_up2 = proba_va[:, class_to_idx[2]]

    p_up_raw = p_up1 + p_up2
    p_down_raw = p_down1 + p_down2
    p_nonflat = p_up_raw + p_down_raw

    with np.errstate(divide="ignore", invalid="ignore"):
        p_up_cond = np.where(p_nonflat > 1e-12, p_up_raw / p_nonflat, 0.5)
        p_down_cond = np.where(p_nonflat > 1e-12, p_down_raw / p_nonflat, 0.5)

    p_up_cond_s = pd.Series(p_up_cond, index=X_va.index)
    p_down_cond_s = pd.Series(p_down_cond, index=X_va.index)
    rt_cost = cost.roundtrip_cost_ret

    # Broaden threshold sweep slightly
    thresholds = np.linspace(0.55, 0.85, 31)
    best_t = 0.65
    best_mean = -1e-9  # avoid selecting empty/flat by mistake

    for t in thresholds:
        # Decision masks
        up_mask = p_up_cond_s >= t
        down_mask = p_down_cond_s >= t

        # Direction decision
        signal = np.where(up_mask & (p_up_cond_s >= p_down_cond_s), 1, 0)
        signal = np.where(down_mask & (p_down_cond_s > p_up_cond_s), -1, signal)

        # Confidence
        denom = max(1e-9, 1.0 - float(t))
        conf_up = (p_up_cond_s.values - float(t)) / denom
        conf_down = (p_down_cond_s.values - float(t)) / denom
        conf = np.zeros_like(conf_up)
        conf = np.where(signal == 1, conf_up, conf)
        conf = np.where(signal == -1, conf_down, conf)

        # Eligibility by confidence
        eligible = conf >= float(min_confidence)
        signal = np.where(eligible, signal, 0)

        pnl = np.where(signal == 1, next_ret_va.values - rt_cost,
                       np.where(signal == -1, -next_ret_va.values - rt_cost, 0.0))
        mean_pnl = float(np.mean(pnl))
        if mean_pnl > best_mean:
            best_mean = mean_pnl
            best_t = float(t)

    return float(best_t), float(best_mean)


def train_multilevel_model(
    symbol: str,
    days: int,
    cost: CostModel,
    progress: Optional[Callable[[str, float], None]] = None,
    trials: int = 60,
    backtest_days: int = 0,
) -> Tuple[str, str]:
    """
    Train a multi-level (5-class) LightGBM model and save to data/processed/.
    Uses Optuna Bayesian HPO with time-series cross-validation to maximize mean PnL.
    Falls back to a small random search if Optuna is unavailable.
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
    feats = build_enriched_features(df, symbol)

    report("Labeling (multi-level)", 0.18)
    labeled = build_multi_level_labels(feats, cost)
    X_all, y_all = feature_target_split(labeled)

    # Remove NaN labels
    mask_all = y_all.notna()
    X_all, y_all = X_all[mask_all], y_all[mask_all].astype(int)
    labeled = labeled.loc[X_all.index]

    if len(X_all) < 5000:
        raise RuntimeError("Not enough samples for advanced training. Increase days.")

    # Split out backtest window if requested
    if backtest_days and backtest_days > 0:
        bt_cut = labeled.index.max() - pd.Timedelta(days=int(backtest_days))
        back_mask = labeled.index >= bt_cut
    else:
        back_mask = pd.Series(False, index=labeled.index)

    # Non-backtest indices for HPO + final fit
    idx_train_val = labeled.index[~back_mask].sort_values()
    n_tv = len(idx_train_val)
    if n_tv < 2000:
        raise RuntimeError("Not enough samples after excluding backtest for training/validation.")

    # Time-series CV windows (3 folds default, 10% validation each)
    n_folds = 3
    vlen = max(500, int(n_tv * 0.10))
    folds = []
    for i in range(n_folds):
        va_end = n_tv - i * vlen
        va_start = max(0, va_end - vlen)
        if va_start <= 0:
            break
        idx_va = idx_train_val[va_start:va_end]
        idx_tr = idx_train_val[:va_start]
        if len(idx_tr) < 1500 or len(idx_va) < 400:
            continue
        folds.append((idx_tr, idx_va))
    folds = list(reversed(folds))  # older->newer

    classes = [-2, -1, 0, 1, 2]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Class weights from the latest train split (or all) for stability
    cw = _class_weights(y_all.loc[idx_train_val], classes)
    class_weight_list = [cw[c] for c in classes]

    import lightgbm as lgb

    # Custom PnL evaluation metric (higher is better) - uses fold's validation segment
    def make_pnl_feval(tr_idx: pd.Index, va_idx: pd.Index, next_ret_series: pd.Series):
        next_ret_tr = next_ret_series.loc[tr_idx].values
        next_ret_va = next_ret_series.loc[va_idx].values
        len_tr, len_va = len(next_ret_tr), len(next_ret_va)

        def pnl_feval(y_pred: np.ndarray, dset: "lgb.Dataset"):
            n = dset.num_data()
            if n == len_tr:
                nr = next_ret_tr
            elif n == len_va:
                nr = next_ret_va
            else:
                # This case should not be hit if used correctly with dtrain/dval
                return ("val_mean_pnl", -1e9, True)

            proba = y_pred.reshape(n, 5)
            p_down2 = proba[:, class_to_idx[-2]]
            p_down1 = proba[:, class_to_idx[-1]]
            p_up1 = proba[:, class_to_idx[1]]
            p_up2 = proba[:, class_to_idx[2]]
            p_up_raw = p_up1 + p_up2
            p_down_raw = p_down1 + p_down2
            p_nonflat = p_up_raw + p_down_raw
            with np.errstate(divide="ignore", invalid="ignore"):
                p_up_cond = np.where(p_nonflat > 1e-12, p_up_raw / p_nonflat, 0.5)
            # quick sweep
            thresholds = np.linspace(0.6, 0.75, 11)
            rt_cost = cost.roundtrip_cost_ret
            best = -1e9
            for t in thresholds:
                sig = np.where(p_up_cond >= t, 1, np.where(p_up_cond <= 1 - t, -1, 0))
                pnl = np.where(sig == 1, nr - rt_cost, np.where(sig == -1, -nr - rt_cost, 0.0))
                m = float(np.mean(pnl))
                if m > best:
                    best = m
            return ("val_mean_pnl", best, True)
        return pnl_feval

    # Try Optuna, else fallback to random search
    best_params = None
    best_score = -1e18
    best_model = None
    best_thr = 0.65

    report("Hyperparameter search", 0.22)
    try:
        import optuna  # type: ignore

        def objective(trial: "optuna.Trial") -> float:
            params = dict(
                objective="multiclass",
                num_class=5,
                metric=["multi_logloss"],
                boosting_type="gbdt",
                num_leaves=trial.suggest_int("num_leaves", 64, 256),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                feature_fraction=trial.suggest_float("feature_fraction", 0.6, 0.95),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 0.95),
                bagging_freq=trial.suggest_int("bagging_freq", 1, 6),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 400),
                max_depth=trial.suggest_categorical("max_depth", [-1, 8, 10, 12]),
                lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                min_gain_to_split=trial.suggest_float("min_gain_to_split", 1e-3, 5.0, log=True),
                verbose=-1,
                seed=trial.suggest_int("seed", 1, 10000),
                class_weight=class_weight_list,
                subsample=trial.suggest_float("subsample", 0.6, 0.95),
                feature_pre_filter=False,
            )
            fold_scores = []
            for (idx_tr, idx_va) in folds:
                X_tr, y_tr = X_all.loc[idx_tr], y_all.loc[idx_tr].map(class_to_idx)
                X_va, y_va = X_all.loc[idx_va], y_all.loc[idx_va].map(class_to_idx)
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_va, label=y_va)
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=3000,
                    valid_sets=[dtrain, dval],
                    valid_names=["train", "valid"],
                    feval=make_pnl_feval(idx_tr, idx_va, labeled["next_ret"]),
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
                )
                # Evaluate mean pnl on this fold using the live eval function
                thr, mean_pnl = _eval_model_pnl(model, X_va, labeled.loc[idx_va, "next_ret"], class_to_idx, cost, min_confidence=0.20)
                fold_scores.append(mean_pnl)
            return -float(np.mean(fold_scores))  # minimize

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=int(max(5, trials)))
        best_params = study.best_trial.params

        # Train a final model on 85/15 split (kept for threshold tuning) using best params
        split = int(n_tv * 0.85)
        tv_idx_sorted = idx_train_val
        idx_tr_final = tv_idx_sorted[:split]
        idx_va_final = tv_idx_sorted[split:]
        X_tr, y_tr = X_all.loc[idx_tr_final], y_all.loc[idx_tr_final].map(class_to_idx)
        X_va, y_va = X_all.loc[idx_va_final], y_all.loc[idx_va_final].map(class_to_idx)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va)
        params = dict(best_params)
        params.update(dict(objective="multiclass", num_class=5, metric=["multi_logloss"], verbose=-1, class_weight=class_weight_list))
        best_model = lgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            feval=make_pnl_feval(idx_tr_final, idx_va_final, labeled["next_ret"]),
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        best_thr, best_score = _eval_model_pnl(best_model, X_va, labeled.loc[idx_va_final, "next_ret"], class_to_idx, cost, min_confidence=0.20)
    except Exception:
        # Fallback to previous small random search on the 85/15 split
        split = int(n_tv * 0.85)
        tv_idx_sorted = idx_train_val
        idx_tr = tv_idx_sorted[:split]
        idx_va = tv_idx_sorted[split:]
        X_tr, y_tr = X_all.loc[idx_tr], y_all.loc[idx_tr]
        X_va, y_va = X_all.loc[idx_va], y_all.loc[idx_va]
        next_ret_tr = labeled.loc[idx_tr, "next_ret"].values
        next_ret_va = labeled.loc[idx_va, "next_ret"].values
        y_tr_idx = y_tr.map(class_to_idx)
        y_va_idx = y_va.map(class_to_idx)

        dtrain = lgb.Dataset(X_tr, label=y_tr_idx)
        dval = lgb.Dataset(X_va, label=y_va_idx)

        def pnl_feval(y_pred: np.ndarray, dset: "lgb.Dataset"):
            n = dset.num_data()
            if n == len(X_tr):
                nr = next_ret_tr
            elif n == len(X_va):
                nr = next_ret_va
            else:
                return ("val_mean_pnl", -1e9, True)

            proba = y_pred.reshape(n, 5)
            p_down2 = proba[:, class_to_idx[-2]]
            p_down1 = proba[:, class_to_idx[-1]]
            p_up1 = proba[:, class_to_idx[1]]
            p_up2 = proba[:, class_to_idx[2]]
            p_up_raw = p_up1 + p_up2
            p_down_raw = p_down1 + p_down2
            p_nonflat = p_up_raw + p_down_raw
            with np.errstate(divide="ignore", invalid="ignore"):
                p_up_cond = np.where(p_nonflat > 1e-12, p_up_raw / p_nonflat, 0.5)

            # This Series creation is not strictly necessary for the logic but was in the original
            # except block. To minimize changes, we'll keep a similar structure, but ensure it
            # doesn't cause an error by using a generic range index. The `p_up_cond` array is all
            # that's used in the calculation.
            p_up_s = pd.Series(p_up_cond)
            thresholds = np.linspace(0.6, 0.75, 11)
            rt_cost = cost.roundtrip_cost_ret
            best = -1e9
            for t in thresholds:
                sig = np.where(p_up_s >= t, 1, np.where(p_up_s <= 1 - t, -1, 0))
                pnl = np.where(sig == 1, nr - rt_cost,
                               np.where(sig == -1, -nr - rt_cost, 0.0))
                m = float(np.mean(pnl))
                if m > best:
                    best = m
            return ("val_mean_pnl", best, True)

        rng = np.random.RandomState(42)
        best_mean_pnl = -1e9
        trials_ = int(max(1, trials))
        for i in range(trials_):
            params = _sample_params(rng, class_weight_list)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=4000,
                valid_sets=[dtrain, dval],
                valid_names=["train", "valid"],
                feval=pnl_feval,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            )
            thr, mean_pnl = _eval_model_pnl(model, X_va, next_ret_va, class_to_idx, cost, min_confidence=0.20)
            if mean_pnl > best_mean_pnl:
                best_mean_pnl = mean_pnl
                best_model = model
                best_thr = thr
            if progress:
                progress(f"Hyperparameter search ({i+1}/{trials_})", 0.22 + 0.5 * (i + 1) / trials_)

    # Optional small backtest on held-out last `backtest_days`
    backtest_metrics: Dict[str, float] = {}
    if back_mask.any() and best_model is not None:
        report("Backtesting (holdout)", 0.75)
        idx_bt = labeled.index[back_mask]
        X_bt = X_all.loc[idx_bt]
        next_ret_bt = labeled.loc[idx_bt, "next_ret"]
        proba_bt = best_model.predict(X_bt, num_iteration=getattr(best_model, "best_iteration", None))

        p_down2_bt = proba_bt[:, class_to_idx[-2]]
        p_down1_bt = proba_bt[:, class_to_idx[-1]]
        p_up1_bt   = proba_bt[:, class_to_idx[1]]
        p_up2_bt   = proba_bt[:, class_to_idx[2]]

        p_up_raw_bt = p_up1_bt + p_up2_bt
        p_down_raw_bt = p_down1_bt + p_down2_bt
        p_nonflat_bt = p_up_raw_bt + p_down_raw_bt
        with np.errstate(divide="ignore", invalid="ignore"):
            p_up_cond_bt = np.where(p_nonflat_bt > 1e-12, p_up_raw_bt / p_nonflat_bt, 0.5)

        p_up_bt_s = pd.Series(p_up_cond_bt, index=idx_bt)
        t = float(best_thr)
        rt_cost = cost.roundtrip_cost_ret
        signal_bt = np.where(p_up_bt_s > t, 1, np.where(p_up_bt_s < 1 - t, -1, 0))
        pnl_bt = np.where(signal_bt == 1, next_ret_bt - rt_cost, np.where(signal_bt == -1, -next_ret_bt - rt_cost, 0.0))
        df_bt = pd.DataFrame({"prob_up": p_up_bt_s, "signal": signal_bt, "next_ret": next_ret_bt, "pnl": pnl_bt}, index=idx_bt)
        os.makedirs("data/processed", exist_ok=True)
        df_bt.to_csv("data/processed/advanced_backtest.csv", index_label="timestamp")

        trades = int((df_bt["signal"] != 0).sum())
        wins = int(((df_bt["signal"] == 1) & (df_bt["next_ret"] > rt_cost)).sum() + ((df_bt["signal"] == -1) & (df_bt["next_ret"] < -rt_cost)).sum())
        win_rate = float(wins / trades) if trades > 0 else 0.0
        cum_pnl = float(df_bt["pnl"].sum())
        expectancy = float(df_bt["pnl"].mean())
        backtest_metrics = dict(trades=trades, win_rate=win_rate, cum_pnl=cum_pnl, expectancy=expectancy, days=int(backtest_days))

    report("Finalizing model", 0.80)

    # Save model and metadata
    if best_model is None:
        raise RuntimeError("Model training failed to produce a valid model.")

    model_path = "data/processed/advanced_model.txt"
    meta_path = "data/processed/advanced_meta.json"
    best_model.save_model(model_path)
    meta = AdvancedMeta(
        symbol=symbol,
        feature_names=list(X_all.columns),
        threshold=float(best_thr),
        taker_fee_bps=float(cost.taker_fee_bps),
        slippage_bps=float(cost.slippage_bps),
        best_iteration=getattr(best_model, "best_iteration", None),
        classes=classes,
    ).__dict__
    meta["trials"] = int(trials)
    # Persist best HPO params and validation score if available
    try:
        meta["best_params"] = dict(best_params) if best_params is not None else None
    except Exception:
        meta["best_params"] = None
    try:
        meta["val_mean_pnl"] = float(best_score)
    except Exception:
        pass
    if back_mask.any():
        meta["backtest_days"] = int(backtest_days)
        meta["backtest_metrics"] = backtest_metrics

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
    threshold_override: Optional[float] = None,
    extra_features: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Timestamp, float, int, float]:
    """
    Predict on the latest completed minute using the advanced model.
    Returns (timestamp, prob_up, signal_strength in {-2..2}, confidence)

    extra_features: optional dict of additional feature_name -> value (e.g., sentiment_manual)
                    Only applied if the feature exists in the trained model's feature list.
    """
    booster: lgb.Booster = bundle["booster"]
    meta = bundle["meta"]
    feature_names: List[str] = meta["feature_names"]
    # Use override threshold if provided, otherwise fall back to meta
    thr: float = float(meta["threshold"])
    if threshold_override is not None:
        try:
            thr = float(threshold_override)
        except Exception:
            pass

    # Get a recent window straight from the exchange for freshness
    ex = ccxt.binance({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=max(600, min(feature_minutes, 1000)))
    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = raw.set_index("timestamp").sort_index().drop_duplicates()

    # Build enriched features (includes macro/on-chain when available)
    feats = build_enriched_features(raw, symbol)
    if feats.empty:
        raise RuntimeError("Insufficient features for prediction.")
    ts = feats.index[-1]
    X_row = feats.iloc[[-1]].copy()

    # Inject any provided extra features if the model expects them
    if extra_features:
        for k, v in extra_features.items():
            if k in feature_names:
                X_row[k] = float(v)

    # Align columns
    for col in feature_names:
        if col not in X_row.columns:
            X_row[col] = 0.0
    X_row = X_row[feature_names]

    prob_vec = booster.predict(X_row)[0]  # length 5
    prob_up, signal, confidence = _confidence_from_probs(np.array(prob_vec, dtype=float), thr)
    return ts, float(prob_up), int(signal), float(confidence)