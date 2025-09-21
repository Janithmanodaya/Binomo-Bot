import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import ccxt

from src.run_pipeline import (
    CostModel,
    build_labels,
    feature_target_split,
    train_lightgbm,
    tune_threshold_for_pnl,
)
# Use rich multi-timeframe features for live logic
from src.feature_lib import build_rich_features


@dataclass
class LiveConfig:
    symbol: str = "ETH/USDT"
    train_days: int = 7            # days of history to train model
    feature_minutes: int = 2000    # minutes of recent data to compute features online
    default_threshold: float = 0.55
    exchange: str = "binance"


class LiveSignalRunner:
    def __init__(
        self,
        cfg: LiveConfig,
        cost: CostModel,
        on_update: Optional[Callable[[Dict[str, object]], None]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        self.cfg = cfg
        self.cost = cost
        self.on_update = on_update
        self._stop = False
        self._ex = None
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._threshold = cfg.default_threshold
        self._last_pred_time: Optional[pd.Timestamp] = None
        self._pending_eval_time: Optional[pd.Timestamp] = None
        self._pending_prob: Optional[float] = None
        self._sleep = sleep_fn or time.sleep

    def stop(self):
        self._stop = True

    # ------------- Data fetch -------------
    def _exchange(self):
        if self._ex is None:
            if self.cfg.exchange != "binance":
                raise NotImplementedError("Only binance spot is implemented in live mode.")
            self._ex = ccxt.binance({"enableRateLimit": True})
        return self._ex

    def _utc_now_ms(self) -> int:
        return int(pd.Timestamp.now(tz="UTC").value // 1_000_000)

    def _to_local(self, ts: pd.Timestamp) -> pd.Timestamp:
        # Convert a UTC timestamp to Sri Lanka time (Asia/Colombo, UTC+5:30)
        return pd.Timestamp(ts).tz_convert("Asia/Colombo")

    def fetch_recent_minutes(self, minutes: int) -> pd.DataFrame:
        """
        Fetch last N minutes of 1m OHLCV. Uses pagination if required.
        """
        ex = self._exchange()
        timeframe = "1m"
        limit = 1000
        since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes + 2)).value // 1_000_000)
        rows: List[List[float]] = []
        while True:
            ohlcv = ex.fetch_ohlcv(self.cfg.symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            since = last_ts + 60_000
            # respect rate limit
            self._sleep(ex.rateLimit / 1000.0)
            if len(ohlcv) < limit:
                break
        if not rows:
            raise RuntimeError("No OHLCV fetched in live fetch.")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        return df

    def fetch_train_days(self, days: int) -> pd.DataFrame:
        minutes = max(days * 1440, self.cfg.feature_minutes)
        return self.fetch_recent_minutes(minutes)

    # ------------- Train model -------------
    def train_model(self):
        if self.on_update:
            self.on_update({"event": "status", "message": f"Training live model on last {self.cfg.train_days} days..."})
        ohlcv = self.fetch_train_days(self.cfg.train_days)
        # Use multi-timeframe rich features
        feats = build_rich_features(ohlcv)
        labeled = build_labels(feats, self.cost)
        X, y = feature_target_split(labeled)
        # Simple split: last 10% for validation threshold tuning
        n = len(X)
        if n < 2000 or y.notna().sum() < 1000:
            raise RuntimeError("Not enough labeled samples for live training. Increase train_days.")
        split = int(n * 0.9)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_va, y_va = X.iloc[split:], y.iloc[split:]
        model = train_lightgbm(X_tr, y_tr, X_va, y_va)
        # Threshold tuning on validation part
        va_mask_notna = y_va.notna()
        if va_mask_notna.any():
            proba = pd.Series(index=X_va.index, dtype=float)
            proba.loc[va_mask_notna] = model.predict(X_va[va_mask_notna], num_iteration=model.best_iteration)
            t_opt = tune_threshold_for_pnl(proba.dropna(), labeled.loc[proba.dropna().index, "next_ret"], self.cost)
        else:
            t_opt = self.cfg.default_threshold
        self._model = model
        self._feature_names = list(X.columns)
        self._threshold = float(t_opt)
        if self.on_update:
            self.on_update({"event": "model_ready", "threshold": self._threshold})

    # ------------- Predict and evaluate -------------
    def _predict_latest(self) -> Tuple[pd.Timestamp, float, int]:
        """
        Predict on the most recent completed minute.
        Returns (timestamp, prob_up, signal)
        """
        recent = self.fetch_recent_minutes(self.cfg.feature_minutes)
        feats = build_rich_features(recent)
        if feats.empty:
            raise RuntimeError("No features computed in live prediction.")
        # Use latest completed index
        ts = feats.index[-1]
        X_row = feats.iloc[[-1]].copy()
        # Align feature columns to the trained model
        if not self._feature_names:
            raise RuntimeError("Model feature names are not set.")
        for col in self._feature_names:
            if col not in X_row.columns:
                X_row[col] = 0.0
        # Keep only training features (order matters)
        X_row = X_row[self._feature_names]
        # Predict probability of UP
        best_it = getattr(self._model, "best_iteration", None)
        prob = float(self._model.predict(X_row, num_iteration=best_it)[0])
        signal = int(1 if prob > self._threshold else (-1 if prob < 1 - self._threshold else 0))
        return ts, prob, signal

    def _evaluate_previous(self, prev_ts: pd.Timestamp) -> Optional[bool]:
        """
        Evaluate correctness of previous prediction using realized next_ret against cost-aware tau.
        Returns True/False for correct, or None if not enough data yet.
        """
        # Fetch at least the next bar
        now = pd.Timestamp.now(tz="UTC")
        if now <= prev_ts + pd.Timedelta(minutes=1, seconds=5):
            return None  # too soon
        recent = self.fetch_recent_minutes(self.cfg.feature_minutes)
        feats = build_rich_features(recent)
        lbl = build_labels(feats, self.cost)
        if prev_ts not in lbl.index:
            return None
        # Previous decision implied UP if prob>t, DOWN if prob<1-t, else FLAT (None)
        # We stored whether we had a signal via pending_prob and threshold
        if self._pending_prob is None:
            return None
        t = self._threshold
        if self._pending_prob > t:
            decided_up = True
            decided_down = False
        elif self._pending_prob < 1 - t:
            decided_up = False
            decided_down = True
        else:
            return None  # No-op decision (flat), skip evaluation

        nx = lbl.loc[prev_ts, "next_ret"]
        tau = self.cost.roundtrip_cost_ret
        if decided_up:
            correct = bool(nx > tau)
        else:
            correct = bool(nx < -tau)
        return correct

    # ------------- Loop -------------
    def run_loop(self, max_iters: Optional[int] = None):
        if self._model is None:
            self.train_model()

        iters = 0
        self._pending_eval_time = None
        self._pending_prob = None

        while not self._stop:
            try:
                # Evaluate previous prediction if due
                if self._pending_eval_time is not None:
                    res = self._evaluate_previous(self._pending_eval_time)
                    if res is not None:
                        if self.on_update:
                            ts_loc_eval = self._to_local(self._pending_eval_time)
                            self.on_update({"event": "evaluation", "timestamp": str(ts_loc_eval), "correct": res})
                        self._pending_eval_time = None
                        self._pending_prob = None

                # Predict for latest completed minute
                ts, prob, signal = self._predict_latest()
                if self._last_pred_time is None or ts > self._last_pred_time:
                    self._last_pred_time = ts
                    self._pending_eval_time = ts
                    self._pending_prob = prob
                    if self.on_update:
                        confidence = float(2.0 * abs(prob - 0.5))
                        ts_loc = self._to_local(ts)
                        self.on_update({
                            "event": "prediction",
                            "timestamp": str(ts_loc),
                            "prob_up": prob,
                            "confidence": confidence,
                            "signal": int(signal),
                            "threshold": float(self._threshold),
                        })

                # Sleep until next minute boundary
                now = pd.Timestamp.now(tz="UTC")
                next_min = (now.floor("T") + pd.Timedelta(minutes=1))
                sleep_s = max(5.0, (next_min - now).total_seconds())
                self._sleep(sleep_s)
            except Exception as e:
                if self.on_update:
                    self.on_update({"event": "error", "message": str(e)})
                # back off briefly
                self._sleep(5.0)

            iters += 1
            if max_iters is not None and iters >= max_iters:
                break