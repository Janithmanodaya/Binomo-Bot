import os
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, Callable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Ensure project root is on sys.path so `from src...` works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Live components
from src.live_signal import LiveConfig, LiveSignalRunner  # type: ignore

try:
    from src.run_pipeline import (
        CostModel,
        run_pipeline,
        fetch_recent_ohlcv_ccxt,
        build_features,
        build_labels,
        feature_target_split,
        final_feature_names,
        train_final_model_on_all,
    )  # type: ignore
except Exception:
    # As a fallback, ensure ROOT is present (should already be)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import (  # type: ignore
        CostModel,
        run_pipeline,
        fetch_recent_ohlcv_ccxt,
        build_features,
        build_labels,
        feature_target_split,
        final_feature_names,
        train_final_model_on_all,
    )
    from src.run_pipeline import (  # type: ignore
        CostModel,
        run_pipeline,
        fetch_recent_ohlcv_ccxt,
        build_features,
        build_labels,
        feature_target_split,
        final_feature_names,
        train_final_model_on_all,
    )


class StdoutRedirector:
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def write(self, s: str):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class TrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Baseline Trainer (Tk)")
        self.geometry("980x760")

        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._stdout_prev: Optional[object] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Live runner state
        self._live_thread: Optional[threading.Thread] = None
        self._live_runner = None
        self._live_stop = threading.Event()

        # Build UI and start polling logs
        try:
            self._build_ui()
        except Exception as e:
            # If any UI build error occurs, render a minimal message so the window isn't empty
            holder = ttk.Frame(self, padding=12)
            holder.pack(fill=tk.BOTH, expand=True)
            ttk.Label(holder, text=f"UI error: {e}").pack(anchor="w")
        self.after(100, self._poll_stdout)

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(frm)
        nb.pack(fill=tk.BOTH, expand=True)

        # --------- Tab: Backtest ----------
        tab_backtest = ttk.Frame(nb)
        nb.add(tab_backtest, text="Backtest")

        cfg = ttk.LabelFrame(tab_backtest, text="Configuration", padding=10)
        cfg.pack(fill=tk.X)

        self.symbol_var = tk.StringVar(value="ETH/USDT")
        self.days_var = tk.IntVar(value=60)
        self.taker_var = tk.DoubleVar(value=4.0)
        self.slip_var = tk.DoubleVar(value=1.0)
        self.folds_var = tk.IntVar(value=5)
        self.valdays_var = tk.IntVar(value=10)
        self.threshold_var = tk.DoubleVar(value=0.55)

        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label, width=28).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="we", padx=4, pady=3)

        cfg.columnconfigure(1, weight=1)
        add_row(cfg, 0, "Symbol (Binance spot):", ttk.Entry(cfg, textvariable=self.symbol_var))
        add_row(cfg, 1, "Lookback days:", ttk.Spinbox(cfg, from_=10, to=365, textvariable=self.days_var, increment=5))
        add_row(cfg, 2, "Taker fee per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.taker_var))
        add_row(cfg, 3, "Slippage per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.slip_var))
        add_row(cfg, 4, "Walk-forward folds:", ttk.Spinbox(cfg, from_=1, to=20, textvariable=self.folds_var))
        add_row(cfg, 5, "Validation days per fold:", ttk.Spinbox(cfg, from_=1, to=60, textvariable=self.valdays_var))
        add_row(cfg, 6, "Default decision threshold:", ttk.Spinbox(cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.threshold_var))

        btns = ttk.Frame(tab_backtest)
        btns.pack(fill=tk.X, pady=(8, 4))
        self.start_btn = ttk.Button(btns, text="Run Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.open_out_btn = ttk.Button(btns, text="Open predictions.csv", command=self.open_output, state=tk.DISABLED)
        self.open_out_btn.pack(side=tk.LEFT, padx=(8, 0))

        prog = ttk.Frame(tab_backtest)
        prog.pack(fill=tk.X, pady=(4, 8))
        self.status_var = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(prog, textvariable=self.status_var)
        self.status_lbl.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, expand=True, padx=(8, 0))

        logf = ttk.LabelFrame(tab_backtest, text="Logs", padding=6)
        logf.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(logf, wrap="word", height=25)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=sb.set)

        self.footer = ttk.Label(tab_backtest, text="Predictions: (not generated yet)")
        self.footer.pack(fill=tk.X, pady=(6, 0))

        # --------- Tab: Live ----------
        tab_live = ttk.Frame(nb)
        nb.add(tab_live, text="Live")

        live_cfg = ttk.LabelFrame(tab_live, text="Live configuration", padding=10)
        live_cfg.pack(fill=tk.X)

        self.live_train_days = tk.IntVar(value=7)
        self.live_feat_minutes = tk.IntVar(value=2000)
        self.live_default_thresh = tk.DoubleVar(value=0.55)

        add_row(live_cfg, 0, "Symbol (Binance spot):", ttk.Entry(live_cfg, textvariable=self.symbol_var))
        add_row(live_cfg, 1, "Train days (history):", ttk.Spinbox(live_cfg, from_=2, to=90, textvariable=self.live_train_days))
        add_row(live_cfg, 2, "Feature minutes (recent):", ttk.Spinbox(live_cfg, from_=500, to=5000, increment=100, textvariable=self.live_feat_minutes))
        add_row(live_cfg, 3, "Default decision threshold:", ttk.Spinbox(live_cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.live_default_thresh))

        live_btns = ttk.Frame(tab_live)
        live_btns.pack(fill=tk.X, pady=(8, 4))
        self.live_start = ttk.Button(live_btns, text="Start Live", command=self.start_live)
        self.live_start.pack(side=tk.LEFT)
        self.live_stop = ttk.Button(live_btns, text="Stop Live", command=self.stop_live, state=tk.DISABLED)
        self.live_stop.pack(side=tk.LEFT, padx=(8, 0))

        live_stats = ttk.LabelFrame(tab_live, text="Live status", padding=10)
        live_stats.pack(fill=tk.X)
        self.live_status = tk.StringVar(value="Idle")
        self.live_prob = tk.StringVar(value="—")
        self.live_signal = tk.StringVar(value="—")
        self.live_eval = tk.StringVar(value="—")

        ttk.Label(live_stats, text="Status:", width=18).grid(row=0, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_status).grid(row=0, column=1, sticky="w")
        ttk.Label(live_stats, text="Prob(up):", width=18).grid(row=1, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_prob).grid(row=1, column=1, sticky="w")
        ttk.Label(live_stats, text="Signal:", width=18).grid(row=2, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_signal).grid(row=2, column=1, sticky="w")
        ttk.Label(live_stats, text="Last evaluation:", width=18).grid(row=3, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_eval).grid(row=3, column=1, sticky="w")

        live_logf = ttk.LabelFrame(tab_live, text="Live logs", padding=6)
        live_logf.pack(fill=tk.BOTH, expand=True)
        self.live_log = tk.Text(live_logf, wrap="word", height=20)
        self.live_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(live_logf, orient="vertical", command=self.live_log.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_log.configure(yscrollcommand=lsb.set)

    def _append_log(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _append_live_log(self, text: str):
        self.live_log.insert(tk.END, text)
        self.live_log.see(tk.END)

    def _poll_stdout(self):
        try:
            while True:
                s = self._stdout_queue.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.after(100, self._poll_stdout)

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(frm)
        nb.pack(fill=tk.BOTH, expand=True)

        # --------- Tab: Backtest ----------
        tab_backtest = ttk.Frame(nb)
        nb.add(tab_backtest, text="Backtest")

        cfg = ttk.LabelFrame(tab_backtest, text="Configuration", padding=10)
        cfg.pack(fill=tk.X)

        self.symbol_var = tk.StringVar(value="ETH/USDT")
        self.days_var = tk.IntVar(value=60)
        self.taker_var = tk.DoubleVar(value=4.0)
        self.slip_var = tk.DoubleVar(value=1.0)
        self.folds_var = tk.IntVar(value=5)
        self.valdays_var = tk.IntVar(value=10)
        self.threshold_var = tk.DoubleVar(value=0.55)

        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label, width=28).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="we", padx=4, pady=3)

        cfg.columnconfigure(1, weight=1)
        add_row(cfg, 0, "Symbol (Binance spot):", ttk.Entry(cfg, textvariable=self.symbol_var))
        add_row(cfg, 1, "Lookback days:", ttk.Spinbox(cfg, from_=10, to=365, textvariable=self.days_var, increment=5))
        add_row(cfg, 2, "Taker fee per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.taker_var))
        add_row(cfg, 3, "Slippage per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.slip_var))
        add_row(cfg, 4, "Walk-forward folds:", ttk.Spinbox(cfg, from_=1, to=20, textvariable=self.folds_var))
        add_row(cfg, 5, "Validation days per fold:", ttk.Spinbox(cfg, from_=1, to=60, textvariable=self.valdays_var))
        add_row(cfg, 6, "Default decision threshold:", ttk.Spinbox(cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.threshold_var))

        btns = ttk.Frame(tab_backtest)
        btns.pack(fill=tk.X, pady=(8, 4))
        self.start_btn = ttk.Button(btns, text="Run Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.open_out_btn = ttk.Button(btns, text="Open predictions.csv", command=self.open_output, state=tk.DISABLED)
        self.open_out_btn.pack(side=tk.LEFT, padx=(8, 0))

        prog = ttk.Frame(tab_backtest)
        prog.pack(fill=tk.X, pady=(4, 8))
        self.status_var = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(prog, textvariable=self.status_var)
        self.status_lbl.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, expand=True, padx=(8, 0))

        logf = ttk.LabelFrame(tab_backtest, text="Logs", padding=6)
        logf.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(logf, wrap="word", height=25)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=sb.set)

        self.footer = ttk.Label(tab_backtest, text="Predictions: (not generated yet)")
        self.footer.pack(fill=tk.X, pady=(6, 0))

        # --------- Tab: Live ----------
        tab_live = ttk.Frame(nb)
        nb.add(tab_live, text="Live")

        live_cfg = ttk.LabelFrame(tab_live, text="Live configuration", padding=10)
        live_cfg.pack(fill=tk.X)

        self.live_train_days = tk.IntVar(value=7)
        self.live_feat_minutes = tk.IntVar(value=2000)
        self.live_default_thresh = tk.DoubleVar(value=0.55)

        add_row(live_cfg, 0, "Symbol (Binance spot):", ttk.Entry(live_cfg, textvariable=self.symbol_var))
        add_row(live_cfg, 1, "Train days (history):", ttk.Spinbox(live_cfg, from_=2, to=90, textvariable=self.live_train_days))
        add_row(live_cfg, 2, "Feature minutes (recent):", ttk.Spinbox(live_cfg, from_=500, to=5000, increment=100, textvariable=self.live_feat_minutes))
        add_row(live_cfg, 3, "Default decision threshold:", ttk.Spinbox(live_cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.live_default_thresh))

        live_btns = ttk.Frame(tab_live)
        live_btns.pack(fill=tk.X, pady=(8, 4))
        self.live_start = ttk.Button(live_btns, text="Start Live", command=self.start_live)
        self.live_start.pack(side=tk.LEFT)
        self.live_stop = ttk.Button(live_btns, text="Stop Live", command=self.stop_live, state=tk.DISABLED)
        self.live_stop.pack(side=tk.LEFT, padx=(8, 0))

        live_stats = ttk.LabelFrame(tab_live, text="Live status", padding=10)
        live_stats.pack(fill=tk.X)
        self.live_status = tk.StringVar(value="Idle")
        self.live_prob = tk.StringVar(value="—")
        self.live_signal = tk.StringVar(value="—")
        self.live_eval = tk.StringVar(value="—")

        ttk.Label(live_stats, text="Status:", width=18).grid(row=0, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_status).grid(row=0, column=1, sticky="w")
        ttk.Label(live_stats, text="Prob(up):", width=18).grid(row=1, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_prob).grid(row=1, column=1, sticky="w")
        ttk.Label(live_stats, text="Signal:", width=18).grid(row=2, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_signal).grid(row=2, column=1, sticky="w")
        ttk.Label(live_stats, text="Last evaluation:", width=18).grid(row=3, column=0, sticky="w")
        ttk.Label(live_stats, textvariable=self.live_eval).grid(row=3, column=1, sticky="w")

        live_logf = ttk.LabelFrame(tab_live, text="Live logs", padding=6)
        live_logf.pack(fill=tk.BOTH, expand=True)
        self.live_log = tk.Text(live_logf, wrap="word", height=20)
        self.live_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(live_logf, orient="vertical", command=self.live_log.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_log.configure(yscrollcommand=lsb.set)

    def _append_log(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _append_live_log(self, text: str):
        self.live_log.insert(tk.END, text)
        self.live_log.see(tk.END)

    def _poll_stdout(self):
        try:
            while True:
                s = self._stdout_queue.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.after(100, self._poll_stdout)

    # ---------------- Backtest actions ----------------
    def start_training(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Training", "Training is already running.")
            return

        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self.open_out_btn.configure(state=tk.DISABLED)
        self._stop_flag.clear()
        self.footer.configure(text="Predictions: (running)")

        self._stdout_prev = sys.stdout
        sys.stdout = StdoutRedirector(self._stdout_queue)  # type: ignore

        self._worker = threading.Thread(target=self._run_training_thread, daemon=True)
        self._worker.start()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

    def stop_training(self):
        if self._worker and self._worker.is_alive():
            self._stop_flag.set()
            self._append_log("\n[User] Stop requested. Current epoch/fold will finish then stop.\n")
        else:
            self._append_log("\nNo active training to stop.\n")

    def _progress_cb(self, stage: str, p: float):
        self.status_var.set(f"{stage} ... {int(p*100)}%")
        self.progress["value"] = int(max(0, min(100, p * 100)))

    def _fold_cb(self, fold: int, metrics: Dict[str, float]):
        self._append_log(
            f"\nFold {fold} metrics: "
            f"AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}, "
            f"Sharpe={metrics['sharpe']:.2f}, PF={metrics['profit_factor']:.2f}, "
            f"Trades={metrics['trades']}\n"
        )

    def _run_training_thread(self):
        out_csv = None
        try:
            cost = CostModel(taker_fee_bps=float(self.taker_var.get()), slippage_bps=float(self.slip_var.get()))
            results, summary, out_csv = run_pipeline(
                symbol=self.symbol_var.get(),
                days=int(self.days_var.get()),
                cost=cost,
                folds=int(self.folds_var.get()),
                val_days=int(self.valdays_var.get()),
                default_prob_threshold=float(self.threshold_var.get()),
                progress_callback=self._progress_cb,
                fold_callback=self._fold_cb,
            )
            self._append_log("\nTraining complete.\n")
            self.status_var.set("Done")
            self.progress["value"] = 100
            if out_csv and os.path.exists(out_csv):
                self.footer.configure(text=f"Predictions: {out_csv}")
                self.open_out_btn.configure(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._append_log(f"\nError: {e}\n")
            self.status_var.set("Error")
        finally:
            if self._stdout_prev is not None:
                sys.stdout = self._stdout_prev  # type: ignore
                self._stdout_prev = None
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    def open_output(self):
        text = self.footer.cget("text")
        if "Predictions:" in text:
            path = text.split("Predictions:", 1)[1].strip()
            if path and os.path.exists(path):
                folder = os.path.abspath(os.path.dirname(path))
                if sys.platform.startswith("win"):
                    os.startfile(folder)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')
            else:
                messagebox.showinfo("Open", "Output file not found.")
        else:
            messagebox.showinfo("Open", "No output yet.")

    # ---------------- Live actions ----------------
    def _on_live_update(self, msg: Dict[str, object]):
        evt = msg.get("event")
        if evt == "status":
            self.live_status.set(str(msg.get("message")))
            self._append_live_log(str(msg.get("message")) + "\n")
        elif evt == "model_ready":
            self.live_status.set(f"Model ready (threshold={msg.get('threshold'):.2f})")
            self._append_live_log(f"Model ready. Threshold={msg.get('threshold'):.2f}\n")
        elif evt == "prediction":
            ts = msg.get("timestamp")
            prob = float(msg.get("prob_up")) if msg.get("prob_up") is not None else float("nan")
            sig = int(msg.get("signal")) if msg.get("signal") is not None else 0
            self.live_prob.set(f"{prob:.3f} @ {ts}")
            sig_str = "LONG" if sig == 1 else ("SHORT" if sig == -1 else "FLAT")
            self.live_signal.set(sig_str)
            self._append_live_log(f"Prediction {ts}: prob_up={prob:.3f}, signal={sig_str}\n")
        elif evt == "evaluation":
            ts = msg.get("timestamp")
            correct = bool(msg.get("correct"))
            self.live_eval.set(f"{'Correct' if correct else 'Wrong'} for {ts}")
            self._append_live_log(f"Evaluation for {ts}: {'Correct' if correct else 'Wrong'}\n")
        elif evt == "error":
            self._append_live_log(f"Error: {msg.get('message')}\n")
        else:
            # generic
            self._append_live_log(str(msg) + "\n")

    def start_live(self):
        if self._live_thread and self._live_thread.is_alive():
            messagebox.showinfo("Live", "Live runner already active.")
            return
        self.live_status.set("Initializing...")
        self._live_stop.clear()

        cost = CostModel(taker_fee_bps=float(self.taker_var.get()), slippage_bps=float(self.slip_var.get()))
        cfg = LiveConfig(
            symbol=self.symbol_var.get(),
            train_days=int(self.live_train_days.get()),
            feature_minutes=int(self.live_feat_minutes.get()),
            default_threshold=float(self.live_default_thresh.get()),
        )
        self._live_runner = LiveSignalRunner(cfg, cost, on_update=self._on_live_update)
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()
        self.live_start.configure(state=tk.DISABLED)
        self.live_stop.configure(state=tk.NORMAL)

    def _live_loop(self):
        try:
            if self._live_runner:
                self._live_runner.run_loop()
        except Exception as e:
            self._append_live_log(f"Live error: {e}\n")
            self.live_status.set("Error")
        finally:
            self.live_start.configure(state=tk.NORMAL)
            self.live_stop.configure(state=tk.DISABLED)

    def stop_live(self):
        if self._live_runner:
            self._live_runner.stop()
        self.live_status.set("Stopping...")

def main():
    app = TrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
