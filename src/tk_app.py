import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `from src...` works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Live components (basic)
from src.live_signal import LiveConfig, LiveSignalRunner  # type: ignore

# Cost model
try:
    from src.run_pipeline import CostModel  # type: ignore
except Exception:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import CostModel  # type: ignore

# Advanced trainer components
from src.advanced_trainer import (  # type: ignore
    train_multilevel_model,
    load_advanced_bundle,
    predict_latest as adv_predict_latest,
)


class LiveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Live Signal (Tk)")
        self.geometry("1100x820")

        # State
        self._live_thread: Optional[threading.Thread] = None
        self._live_runner = None
        self._stop_event = threading.Event()

        # Advanced model bundle (if loaded or trained)
        self._adv_bundle: Optional[Dict] = None
        # Track model file mtimes for hot-reload/versioning
        self._adv_model_mtime: Optional[float] = None
        self._adv_meta_mtime: Optional[float] = None

        # Virtual orders
        self._min_conf_var = tk.DoubleVar(value=0.80)  # confidence filter default 0.80
        self._pending_trade: Optional[Dict] = None  # keyed by last prediction
        self._trades: List[Dict] = []

        # Running PnL metrics
        self._cum_pnl: float = 0.0
        self._cum_pnl_weighted: float = 0.0

        # Build UI
        try:
            self._build_ui()
        except Exception as e:
            holder = ttk.Frame(self, padding=12)
            holder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(holder, text=f"UI error: {e}").pack(anchor="w")

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Live configuration
        cfg = ttk.LabelFrame(frm, text="Live configuration", padding=10)
        cfg.pack(fill=tk.X)

        self.symbol_var = tk.StringVar(value="ETH/USDT")
        self.taker_var = tk.DoubleVar(value=4.0)
        self.slip_var = tk.DoubleVar(value=1.0)

        self.live_train_days = tk.IntVar(value=60)
        self.live_feat_minutes = tk.IntVar(value=2000)
        self.live_default_thresh = tk.DoubleVar(value=0.55)

        # Advanced training controls
        self.adv_trials_var = tk.IntVar(value=60)
        # Parallel jobs for HPO/training (default: half of CPUs capped at 8)
        try:
            import os as _os
            _cpu = _os.cpu_count() or 1
            _default_jobs = max(1, min(8, _cpu // 2 if _cpu >= 2 else 1))
        except Exception:
            _default_jobs = 2
        self.adv_jobs_var = tk.IntVar(value=_default_jobs)
        self.adv_backtest_days_var = tk.IntVar(value=7)
        # Advanced live overrides
        self.adv_threshold_var = tk.DoubleVar(value=0.00)  # 0 -> use meta threshold

        self.model_type = tk.StringVar(value="Advanced")  # "Basic" or "Advanced"

        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label, width=28).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="we", padx=4, pady=3)

        cfg.columnconfigure(1, weight=1)
        add_row(cfg, 0, "Model type:", ttk.Combobox(cfg, textvariable=self.model_type, values=["Advanced", "Basic"], state="readonly"))
        add_row(cfg, 1, "Symbol (Binance spot):", ttk.Entry(cfg, textvariable=self.symbol_var))
        add_row(cfg, 2, "Taker fee per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.taker_var))
        add_row(cfg, 3, "Slippage per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.slip_var))
        add_row(cfg, 4, "Train days (history):", ttk.Spinbox(cfg, from_=2, to=180, textvariable=self.live_train_days))
        add_row(cfg, 5, "Feature minutes (recent):", ttk.Spinbox(cfg, from_=500, to=5000, increment=100, textvariable=self.live_feat_minutes))
        add_row(cfg, 6, "Default decision threshold:", ttk.Spinbox(cfg, from_=0.50, to=0.90, increment=0.01, textvariable=self.live_default_thresh))
        add_row(cfg, 7, "Min confidence (filter+trades):", ttk.Spinbox(cfg, from_=0.00, to=1.00, increment=0.01, textvariable=self._min_conf_var))
        # Break out widget creation to avoid very long argument lists on one line
        spin_trials = ttk.Spinbox(cfg, from_=1, to=200, increment=1, textvariable=self.adv_trials_var)
        add_row(cfg, 8, "Advanced trials (HPO):", spin_trials)

        spin_bt_days = ttk.Spinbox(cfg, from_=0, to=60, increment=1, textvariable=self.adv_backtest_days_var)
        add_row(cfg, 9, "Backtest holdout days:", spin_bt_days)

        spin_jobs = ttk.Spinbox(cfg, from_=1, to=32, increment=1, textvariable=self.adv_jobs_var)
        add_row(cfg, 10, "Advanced parallel jobs:", spin_jobs)

        spin_thr = ttk.Spinbox(cfg, from_=0.00, to=0.90, increment=0.01, textvariable=self.adv_threshold_var)
        add_row(cfg, 11, "Advanced live threshold (0=auto):", spin_thr)

        # Controls
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(8, 4))
        self.live_start = ttk.Button(btns, text="Start Live", command=self.start_live)
        self.live_start.pack(side=tk.LEFT)
        self.live_stop = ttk.Button(btns, text="Stop Live", command=self.stop_live, state=tk.DISABLED)
        self.live_stop.pack(side=tk.LEFT, padx=(8, 0))
        self.save_trades_btn = ttk.Button(btns, text="Save Trades CSV", command=self.save_trades_csv)
        self.save_trades_btn.pack(side=tk.RIGHT)

        # Advanced model actions
        adv_btns = ttk.Frame(frm)
        adv_btns.pack(fill=tk.X, pady=(0, 8))
        self.train_adv_btn = ttk.Button(adv_btns, text="Train Advanced Now", command=self.train_advanced_now)
        self.train_adv_btn.pack(side=tk.LEFT)
        self.load_adv_btn = ttk.Button(adv_btns, text="Load Advanced Model...", command=self.load_advanced_model_dialog)
        self.load_adv_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.dl_hist_btn = ttk.Button(adv_btns, text="Download History", command=self.download_history_now)
        self.dl_hist_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Training progress
        progf = ttk.LabelFrame(frm, text="Training progress", padding=10)
        progf.pack(fill=tk.X)
        self.train_status = tk.StringVar(value="Idle")
        self.train_status_lbl = ttk.Label(progf, textvariable=self.train_status)
        self.train_status_lbl.pack(side=tk.LEFT)
        self.train_progress = ttk.Progressbar(progf, orient="horizontal", mode="determinate", maximum=100, length=300)
        self.train_progress.pack(side=tk.LEFT, padx=(12, 0), fill=tk.X, expand=True)

        # Status
        live_stats = ttk.LabelFrame(frm, text="Live status", padding=10)
        live_stats.pack(fill=tk.X)
        self.live_status = tk.StringVar(value="Idle")
        self.live_prob = tk.StringVar(value="—")
        self.live_signal = tk.StringVar(value="—")
        self.live_eval = tk.StringVar(value="—")
        self.live_pnl = tk.StringVar(value="—")

        ttk.Label(live_stats, text="Status:", width=18).grid(row=0, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_status).grid(row=0, column=1, sticky="w")
        ttk.Label(live_stats, text="Prob(up):", width=18).grid(row=1, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_prob).grid(row=1, column=1, sticky="w")
        ttk.Label(live_stats, text="Signal:", width=18).grid(row=2, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_signal).grid(row=2, column=1, sticky="w")
        ttk.Label(live_stats, text="Last evaluation:", width=18).grid(row=3, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_eval).grid(row=3, column=1, sticky="w")
        ttk.Label(live_stats, text="Cum PnL (w / raw):", width=18).grid(row=4, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_pnl).grid(row=4, column=1, sticky="w")

        # Logs and Trades
        lower = ttk.Panedwindow(frm, orient=tk.HORIZONTAL)
        lower.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        live_logf = ttk.LabelFrame(lower, text="Live logs", padding=6)
        self.live_log = tk.Text(live_logf, wrap="word", height=24)
        self.live_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(live_logf, orient="vertical", command=self.live_log.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_log.configure(yscrollcommand=lsb.set)

        tradesf = ttk.LabelFrame(lower, text="Virtual trades", padding=6)
        cols = ("entry_ts", "side", "entry_price", "exit_ts", "exit_price", "confidence", "result", "pnl", "pnl_w")
        self.trades_view = ttk.Treeview(tradesf, columns=cols, show="headings", height=18)
        for c in cols:
            self.trades_view.heading(c, text=c)
            self.trades_view.column(c, width=110, anchor="center")
        self.trades_view.pack(fill=tk.BOTH, expand=True)

        # Backtest summary table
        backtestf = ttk.LabelFrame(lower, text="Backtest summary", padding=6)
        bt_cols = ("days", "trades", "win_rate", "cum_pnl", "expectancy")
        self.backtest_view = ttk.Treeview(backtestf, columns=bt_cols, show="headings", height=6)
        for c in bt_cols:
            self.backtest_view.heading(c, text=c)
            self.backtest_view.column(c, width=110, anchor="center")
        self.backtest_view.pack(fill=tk.BOTH, expand=True)

        lower.add(live_logf, weight=1)
        lower.add(tradesf, weight=1)
        lower.add(backtestf, weight=0)

    # ---------------- Helpers ----------------
    def _append_live_log(self, text: str):
        self.live_log.insert(tk.END, text)
        self.live_log.see(tk.END)

    def _to_utc_from_local_str(self, ts_local_str: str) -> pd.Timestamp:
        ts = pd.Timestamp(ts_local_str)
        if ts.tzinfo is None or ts.tz is None:
            ts = ts.tz_localize("Asia/Colombo")
        return ts.tz_convert("UTC")

    def _fetch_prices_for_ts(self, symbol: str, ts_utc: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Robustly fetch entry (c0) and next (c1) close prices for a given UTC minute.
        Tries an anchored fetch using 'since' to guarantee inclusion of the desired candles,
        and falls back to a larger recent window if needed.
        """
        try:
            import ccxt
            ex = ccxt.binance({"enableRateLimit": True})
            ts_utc = pd.Timestamp(ts_utc).tz_convert("UTC")
            next_ts = (ts_utc.floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")

            # 1) Try anchored fetch around the target minute
            since_ms = int((ts_utc - pd.Timedelta(minutes=2)).timestamp() * 1000)
            try:
                raw = ex.fetch_ohlcv(symbol, timeframe="1m", since=since_ms, limit=10)
            except Exception:
                raw = []

            def to_df(rows):
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index().drop_duplicates()
                return df

            ok = False
            if raw:
                df = to_df(raw)
                if (ts_utc in df.index) and (next_ts in df.index):
                    ok = True
                    return {"c0": float(df.loc[ts_utc, "close"]), "c1": float(df.loc[next_ts, "close"])}

            # 2) Fallback: fetch a larger recent window
            if not ok:
                raw2 = ex.fetch_ohlcv(symbol, timeframe="1m", limit=120)
                if raw2:
                    df2 = to_df(raw2)
                    if (ts_utc in df2.index) and (next_ts in df2.index):
                        return {"c0": float(df2.loc[ts_utc, "close"]), "c1": float(df2.loc[next_ts, "close"])}

        except Exception:
            pass
        return None

    def _maybe_open_virtual_trade(self, symbol: str, signal: int, confidence: float, ts_local_str: str):
        """
        Open a virtual trade only if we can reliably resolve entry and next prices
        for the prediction timestamp. Otherwise skip and log.
        """
        try:
            if signal == 0 or confidence < float(self._min_conf_var.get()):
                return
            ts_utc = self._to_utc_from_local_str(ts_local_str)
            prices = self._fetch_prices_for_ts(symbol, ts_utc)
            if not prices:
                # Could not fetch matching candles; skip opening this trade
                self._append_live_log(f"Warning: could not resolve entry price for {ts_local_str}; skipping virtual trade.\n")
                return
            entry_price = float(prices["c0"])
            side = "LONG" if signal == 1 else "SHORT"
            self._pending_trade = dict(
                entry_ts=ts_local_str,
                side=side,
                entry_price=entry_price,
                confidence=float(confidence),
            )
        except Exception as e:
            self._append_live_log(f"Virtual trade open error: {e}\n")

    def _close_virtual_trade(self, symbol: str, ts_local_str: str, correct: Optional[bool] = None):
        if not self._pending_trade:
            return
        try:
            entry_ts_local = self._pending_trade["entry_ts"]
            if entry_ts_local != ts_local_str:
                return  # only close when evaluation matches the prediction timestamp
            ts_utc = self._to_utc_from_local_str(ts_local_str)
            prices = self._fetch_prices_for_ts(symbol, ts_utc)
            exit_price = prices["c1"] if prices else float("nan")
            entry_price = float(self._pending_trade["entry_price"])
            side = str(self._pending_trade["side"])
            conf = float(self._pending_trade["confidence"])
            # Compute pnl in log-return terms cost-aware
            tau = 2.0 * (float(self.taker_var.get()) + float(self.slip_var.get())) / 10000.0
            pnl = 0.0
            result = "FLAT"
            if not np.isnan(entry_price) and not np.isnan(float(exit_price)):
                next_ret = float(np.log(float(exit_price)) - np.log(float(entry_price)))
                if side == "LONG":
                    pnl = next_ret - tau
                    result = "WIN" if pnl > 0 else "LOSS"
                elif side == "SHORT":
                    pnl = -next_ret - tau
                    result = "WIN" if pnl > 0 else "LOSS"
            # Prefer correctness flag if provided to set WIN/LOSS label (does not change pnl calc)
            if correct is not None:
                result = "WIN" if bool(correct) else "LOSS" if side in ("LONG", "SHORT") else "FLAT"

            # Confidence-weighted PnL multiplier:
            # Map confidence in [0,1] to weight ~ [0.1, 2.0], with examples:
            # 0.50 -> 0.5x, 0.90 -> 1.5x
            def _weight(c: float) -> float:
                w = 0.5 + 2.5 * (float(c) - 0.5)  # linear map hitting the given examples
                return float(max(0.1, min(2.0, w)))

            pnl_w = float(pnl * _weight(conf))

            trade = dict(
                entry_ts=entry_ts_local,
                side=side,
                entry_price=float(entry_price),
                exit_ts=ts_local_str,
                exit_price=float(exit_price) if prices else float("nan"),
                confidence=conf,
                result=result,
                pnl=float(pnl),
                pnl_w=float(pnl_w),
            )
            self._trades.append(trade)

            # Update cumulative metrics
            self._cum_pnl += float(pnl)
            self._cum_pnl_weighted += float(pnl_w)
            self.live_pnl.set(f"{self._cum_pnl_weighted:.3e} / {self._cum_pnl:.3e}")

            self.trades_view.insert("", tk.END, values=(
                trade["entry_ts"], trade["side"], f"{trade['entry_price']:.6f}",
                trade["exit_ts"], f"{trade['exit_price']:.6f}" if not np.isnan(trade["exit_price"]) else "NaN",
                f"{trade['confidence']:.2f}", trade["result"], f"{trade['pnl']:.3e}", f"{trade['pnl_w']:.3e}"
            ))
        finally:
            self._pending_trade = None

    def save_trades_csv(self):
        try:
            os.makedirs("data/processed", exist_ok=True)
            out = "data/processed/virtual_trades.csv"
            pd.DataFrame(self._trades).to_csv(out, index=False)
            self._append_live_log(f"Saved trades CSV: {out}\n")
        except Exception as e:
            messagebox.showerror("Save Trades", str(e))

    # ---------------- Advanced actions (train/load/history) ----------------
    def train_advanced_now(self):
        symbol = self.symbol_var.get()
        days = int(self.live_train_days.get())
        cost = CostModel(
            taker_fee_bps=float(self.taker_var.get()),
            slippage_bps=float(self.slip_var.get()),
        )
        self.train_adv_btn.configure(state=tk.DISABLED)
        try:
            def on_prog(stage: str, p: float):
                self._update_train_progress(stage, p)
                self.live_status.set(stage)
            self._update_train_progress("Starting training", 0.0)
            train_multilevel_model(
                symbol,
                days,
                cost,
                progress=on_prog,
                trials=int(self.adv_trials_var.get()),
                backtest_days=int(self.adv_backtest_days_var.get()),
            )
            self._adv_bundle = load_advanced_bundle()
            if self._adv_bundle is None:
                raise RuntimeError("Advanced model files not found after training.")
            # Track mtimes for versioning
            try:
                self._adv_model_mtime = os.path.getmtime("data/processed/advanced_model.txt")
                self._adv_meta_mtime = os.path.getmtime("data/processed/advanced_meta.json")
            except Exception:
                self._adv_model_mtime = None
                self._adv_meta_mtime = None
            meta = self._adv_bundle["meta"]
            thr = float(meta.get("threshold", float(self.live_default_thresh.get())))
            self.train_progress["value"] = 100
            self.train_status.set("Training complete")
            self.live_status.set(f"Model ready (threshold={thr:.2f})")
            # Log backtest metrics if present and update table
            bt = meta.get("backtest_metrics")
            # Clear backtest table
            try:
                for item in self.backtest_view.get_children():
                    self.backtest_view.delete(item)
            except Exception:
                pass
            if bt:
                self._append_live_log(
                    "Advanced model trained and loaded.\n"
                    f"- Backtest ({bt.get('days', 0)} days): trades={bt.get('trades', 0)}, "
                    f"win_rate={bt.get('win_rate', 0.0):.3f}, cum_pnl={bt.get('cum_pnl', 0.0):.3e}, "
                    f"expectancy={bt.get('expectancy', 0.0):.3e}\n"
                )
                try:
                    self.backtest_view.insert("", tk.END, values=(
                        bt.get("days", 0),
                        bt.get("trades", 0),
                        f"{bt.get('win_rate', 0.0):.3f}",
                        f"{bt.get('cum_pnl', 0.0):.3e}",
                        f"{bt.get('expectancy', 0.0):.3e}",
                    ))
                except Exception:
                    pass
            else:
                self._append_live_log("Advanced model trained and loaded.\n")
        except Exception as e:
            messagebox.showerror("Advanced Training", str(e))
            self._append_live_log(f"Advanced training error: {e}\n")
            self.live_status.set("Training error")
        finally:
            self.train_adv_btn.configure(state=tk.NORMAL)

    def load_advanced_model_dialog(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select LightGBM model file",
                filetypes=[("LightGBM model", "*.txt"), ("All files", "*.*")]
            )
            if not model_path:
                return
            meta_path = filedialog.askopenfilename(
                title="Select meta JSON file",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")]
            )
            if not meta_path:
                return
            import json
            import lightgbm as lgb  # type: ignore
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            booster = lgb.Booster(model_file=model_path)
            self._adv_bundle = {"booster": booster, "meta": meta}
            # Track mtimes for versioning (best-effort)
            try:
                self._adv_model_mtime = os.path.getmtime(model_path)
                self._adv_meta_mtime = os.path.getmtime(meta_path)
            except Exception:
                self._adv_model_mtime = None
                self._adv_meta_mtime = None
            thr = float(meta.get("threshold", float(self.live_default_thresh.get())))
            self.live_status.set(f"Loaded model (threshold={thr:.2f})")
            self._append_live_log(f"Loaded model:\n- model: {model_path}\n- meta: {meta_path}\n")
            # Populate backtest summary table if available
            bt = meta.get("backtest_metrics")
            try:
                for item in self.backtest_view.get_children():
                    self.backtest_view.delete(item)
            except Exception:
                pass
            if bt:
                try:
                    self.backtest_view.insert("", tk.END, values=(
                        bt.get("days", 0),
                        bt.get("trades", 0),
                        f"{bt.get('win_rate', 0.0):.3f}",
                        f"{bt.get('cum_pnl', 0.0):.3e}",
                        f"{bt.get('expectancy', 0.0):.3e}",
                    ))
                except Exception:
                    pass
        except Exception as e:
            messagebox.showerror("Load Model", str(e))

    def download_history_now(self):
        try:
            symbol = self.symbol_var.get()
            days = int(self.live_train_days.get())
            self.live_status.set(f"Downloading {days} days for {symbol} ...")
            self._append_live_log(f"Downloading history: {symbol}, {days} days\n")
            import ccxt
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
            os.makedirs("data/raw", exist_ok=True)
            suffix = f"{symbol.replace('/', '')}_{days}d_1m.parquet"
            out_path = os.path.join("data", "raw", suffix)
            df.to_parquet(out_path)
            self._append_live_log(f"Saved history to: {out_path} ({len(df):,} rows)\n")
            self.live_status.set("Download complete")
        except Exception as e:
            messagebox.showerror("Download History", str(e))
            self.live_status.set("Download error")

    # ---------------- Live actions ----------------
    def _on_live_update(self, msg: Dict[str, object]):
        evt = msg.get("event")
        if evt == "status":
            self.live_status.set(str(msg.get("message")))
            self._append_live_log(str(msg.get("message")) + "\n")
        elif evt == "model_ready":
            thr = msg.get("threshold")
            try:
                thr_f = float(thr) if thr is not None else float("nan")
                self.live_status.set(f"Model ready (threshold={thr_f:.2f})")
                self._append_live_log(f"Model ready. Threshold={thr_f:.2f}\n")
            except Exception:
                self.live_status.set("Model ready")
                self._append_live_log("Model ready.\n")
        elif evt == "prediction":
            ts_local = str(msg.get("timestamp"))
            prob = float(msg.get("prob_up")) if msg.get("prob_up") is not None else float("nan")
            conf = float(msg.get("confidence")) if msg.get("confidence") is not None else 0.0
            sig = int(msg.get("signal")) if msg.get("signal") is not None else 0
            min_conf = float(self._min_conf_var.get())
            self.live_prob.set(f"{prob:.3f} (conf {conf:.2f}) @ {ts_local}")
            if conf < min_conf or sig == 0:
                sig_str = "FLAT (filtered)" if sig != 0 else "FLAT"
                eff_sig = 0
            else:
                sig_str = "LONG" if sig == 1 else "SHORT"
                eff_sig = sig
            self.live_signal.set(sig_str)
            self._append_live_log(f"Prediction {ts_local}: prob_up={prob:.3f}, confidence={conf:.2f}, signal={sig_str}\n")
            # Virtual order: maybe open (only if passes filter)
            self._maybe_open_virtual_trade(self.symbol_var.get(), eff_sig, conf, ts_local)
        elif evt == "evaluation":
            ts_local = str(msg.get("timestamp"))
            correct = msg.get("correct")
            self.live_eval.set(f"{'Correct' if correct else 'Wrong'} for {ts_local}")
            self._append_live_log(f"Evaluation for {ts_local}: {'Correct' if correct else 'Wrong'}\n")
            # Close virtual trade if one pending for this timestamp
            self._close_virtual_trade(self.symbol_var.get(), ts_local, correct=bool(correct) if correct is not None else None)
            # Persist live feedback for (future) online learning
            try:
                os.makedirs("data/feedback", exist_ok=True)
                path = "data/feedback/live_feedback.csv"
                # Record minimal schema; features will be rebuilt during retraining
                rec = {
                    "timestamp_local": ts_local,
                    "symbol": self.symbol_var.get(),
                    "min_conf": float(self._min_conf_var.get()),
                    "taker_fee_bps": float(self.taker_var.get()),
                    "slippage_bps": float(self.slip_var.get()),
                    "correct": bool(correct) if correct is not None else None,
                }
                # Also attach last prediction line if available from UI state
                rec["last_prob_up"] = None
                rec["last_confidence"] = None
                rec["last_signal"] = self.live_signal.get()
                try:
                    # Parse from live_prob string
                    lp = self.live_prob.get()
                    # Format e.g. "0.612 (conf 0.85) @ 2024-..."; extract numbers safely
                    import re
                    m = re.search(r"([0-9.]+) \\(conf ([0-9.]+)\\)", lp)
                    if m:
                        rec["last_prob_up"] = float(m.group(1))
                        rec["last_confidence"] = float(m.group(2))
                except Exception:
                    pass
                df_one = pd.DataFrame([rec])
                if os.path.exists(path):
                    df_one.to_csv(path, mode="a", header=False, index=False)
                else:
                    df_one.to_csv(path, index=False)
            except Exception:
                pass
        elif evt == "error":
            self._append_live_log(f"Error: {msg.get('message')}\n")
        else:
            self._append_live_log(str(msg) + "\n")

    def _update_train_progress(self, stage: str, p: float):
        self.train_status.set(f"{stage} ... {int(p*100)}%")
        self.train_progress["value"] = int(max(0, min(100, p * 100)))

    def start_live(self):
        if self._live_thread and self._live_thread.is_alive():
            messagebox.showinfo("Live", "Live runner already active.")
            return
        self.live_status.set("Initializing...")
        self._stop_event.clear()
        self._pending_trade = None
        self._trades.clear()
        self._cum_pnl = 0.0
        self._cum_pnl_weighted = 0.0
        self.live_pnl.set("0.000e+00 / 0.000e+00")
        for item in self.trades_view.get_children():
            self.trades_view.delete(item)

        if self.model_type.get() == "Advanced":
            self._live_thread = threading.Thread(target=self._advanced_loop, daemon=True)
        else:
            self._live_thread = threading.Thread(target=self._basic_loop, daemon=True)
        self._live_thread.start()
        self.live_start.configure(state=tk.DISABLED)
        self.live_stop.configure(state=tk.NORMAL)

    def _basic_loop(self):
        try:
            cost = CostModel(
                taker_fee_bps=float(self.taker_var.get()),
                slippage_bps=float(self.slip_var.get()),
            )
            cfg = LiveConfig(
                symbol=self.symbol_var.get(),
                train_days=int(self.live_train_days.get()),
                feature_minutes=int(self.live_feat_minutes.get()),
                default_threshold=float(self.live_default_thresh.get()),
            )
            self._live_runner = LiveSignalRunner(cfg, cost, on_update=self._on_live_update)
            self._live_runner.run_loop()
        except Exception as e:
            self._append_live_log(f"Live error: {e}\n")
            self.live_status.set("Error")
        finally:
            self.live_start.configure(state=tk.NORMAL)
            self.live_stop.configure(state=tk.DISABLED)

    def _advanced_loop(self):
        try:
            symbol = self.symbol_var.get()
            days = int(self.live_train_days.get())
            cost = CostModel(
                taker_fee_bps=float(self.taker_var.get()),
                slippage_bps=float(self.slip_var.get()),
            )

            # Train or use loaded advanced model with progress
            if self._adv_bundle is None:
                def on_prog(stage: str, p: float):
                    self._update_train_progress(stage, p)
                    self.live_status.set(stage)

                self._update_train_progress("Starting training", 0.0)
                try:
                    train_multilevel_model(
                        symbol,
                        days,
                        cost,
                        progress=on_prog,
                        trials=int(self.adv_trials_var.get()),
                        backtest_days=int(self.adv_backtest_days_var.get()),
                        n_jobs=int(self.adv_jobs_var.get()),
                    )
                except Exception as e:
                    self._append_live_log(f"Advanced training error: {e}\n")
                    self.live_status.set("Training error")
                    return

                self._adv_bundle = load_advanced_bundle()
                if self._adv_bundle is None:
                    self._append_live_log("No advanced model found after training.\n")
                    self.live_status.set("Error")
                    return
                self.train_progress["value"] = 100
                self.train_status.set("Training complete")
            else:
                self._append_live_log("Using loaded advanced model bundle.\n")

            bundle = self._adv_bundle
            # Capture current mtimes if available
            try:
                self._adv_model_mtime = os.path.getmtime("data/processed/advanced_model.txt")
                self._adv_meta_mtime = os.path.getmtime("data/processed/advanced_meta.json")
            except Exception:
                self._adv_model_mtime = None
                self._adv_meta_mtime = None
            meta = bundle["meta"]
            thr = float(meta.get("threshold", float(self.live_default_thresh.get())))
            self.live_status.set(f"Model ready (threshold={thr:.2f})")
            bt = meta.get("backtest_metrics")
            # Clear and update backtest table
            try:
                for item in self.backtest_view.get_children():
                    self.backtest_view.delete(item)
            except Exception:
                pass
            if bt:
                self._append_live_log(
                    f"Backtest ({bt.get('days', 0)} days): trades={bt.get('trades', 0)}, "
                    f"win_rate={bt.get('win_rate', 0.0):.3f}, cum_pnl={bt.get('cum_pnl', 0.0):.3e}, "
                    f"expectancy={bt.get('expectancy', 0.0):.3e}\n"
                )
                try:
                    self.backtest_view.insert("", tk.END, values=(
                        bt.get("days", 0),
                        bt.get("trades", 0),
                        f"{bt.get('win_rate', 0.0):.3f}",
                        f"{bt.get('cum_pnl', 0.0):.3e}",
                        f"{bt.get('expectancy', 0.0):.3e}",
                    ))
                except Exception:
                    pass

            # Live prediction loop
            while not self._stop_event.is_set():
                # Hot-reload model if newer files are available (model versioning)
                try:
                    model_path = "data/processed/advanced_model.txt"
                    meta_path = "data/processed/advanced_meta.json"
                    if os.path.exists(model_path) and os.path.exists(meta_path):
                        m_m = os.path.getmtime(model_path)
                        m_j = os.path.getmtime(meta_path)
                        if (self._adv_model_mtime is None) or (self._adv_meta_mtime is None) or (m_m > self._adv_model_mtime) or (m_j > self._adv_meta_mtime):
                            new_bundle = load_advanced_bundle()
                            if new_bundle is not None:
                                bundle = new_bundle
                                meta = bundle["meta"]
                                thr = float(meta.get("threshold", float(self.live_default_thresh.get())))
                                self._adv_bundle = bundle
                                self._adv_model_mtime = m_m
                                self._adv_meta_mtime = m_j
                                self._on_live_update({"event": "model_ready", "threshold": thr})
                except Exception:
                    pass

                thr_override = float(self.adv_threshold_var.get())
                ts, prob_up, strength, conf = adv_predict_latest(
                    symbol=symbol,
                    feature_minutes=int(self.live_feat_minutes.get()),
                    bundle=bundle,
                    threshold_override=thr_override if thr_override > 0.0 else None,
                )
                # Use the advanced model's strength and confidence directly.
                # strength ∈ {-2,-1,0,1,2} -> directional signal {-1,0,1}
                sig = 1 if int(strength) > 0 else (-1 if int(strength) < 0 else 0)
                conf = float(conf)

                # Emit prediction
                ts_local_str = str(pd.Timestamp(ts).tz_convert("Asia/Colombo"))
                self._on_live_update({
                    "event": "prediction",
                    "timestamp": ts_local_str,
                    "prob_up": prob_up,
                    "confidence": conf,
                    "signal": sig,
                    "threshold": thr,
                })

                # Evaluate after next bar closes
                next_ts = (pd.Timestamp(ts).floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")
                sleep_s = max(2.0, (next_ts - pd.Timestamp.now(tz="UTC")).total_seconds() + 2.0)
                # Allow responsive stopping during sleep
                end_time = time.time() + sleep_s
                while time.time() < end_time:
                    if self._stop_event.is_set():
                        break
                    time.sleep(min(0.5, end_time - time.time()))
                if self._stop_event.is_set():
                    break

                # Fetch prices and evaluate correctness cost-aware
                import ccxt
                ex = ccxt.binance({"enableRateLimit": True})
                raw2_rows = ex.fetch_ohlcv(symbol, timeframe="1m", limit=5)
                raw2 = pd.DataFrame(raw2_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                raw2["timestamp"] = pd.to_datetime(raw2["timestamp"], unit="ms", utc=True)
                raw2 = raw2.set_index("timestamp").sort_index().drop_duplicates()
                ts_utc = pd.Timestamp(ts).tz_convert("UTC")
                if (ts_utc in raw2.index) and (next_ts in raw2.index):
                    c0 = float(raw2.loc[ts_utc, "close"])
                    c1 = float(raw2.loc[next_ts, "close"])
                    next_ret = float(np.log(c1) - np.log(c0))
                    tau = 2.0 * (float(cost.taker_fee_bps) + float(cost.slippage_bps)) / 10000.0
                    if sig == 1:
                        correct = bool(next_ret > tau)
                    elif sig == -1:
                        correct = bool(next_ret < -tau)
                    else:
                        correct = bool(abs(next_ret) <= tau)
                    self._on_live_update({
                        "event": "evaluation",
                        "timestamp": ts_local_str,
                        "correct": correct,
                    })
        except Exception as e:
            self._append_live_log(f"Live error: {e}\n")
            self.live_status.set("Error")
        finally:
            self.live_start.configure(state=tk.NORMAL)
            self.live_stop.configure(state=tk.DISABLED)

    def stop_live(self):
        # Signal any loop to stop
        self._stop_event.set()
        # Stop basic runner if used
        try:
            if self._live_runner:
                self._live_runner.stop()
        except Exception:
            pass
        self.live_status.set("Stopping...")


def main():
    app = LiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
