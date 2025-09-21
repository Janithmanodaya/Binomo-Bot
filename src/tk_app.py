import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import ttk, messagebox

# Ensure project root is on sys.path so `from src...` works regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Live components only
from src.live_signal import LiveConfig, LiveSignalRunner  # type: ignore

try:
    from src.run_pipeline import CostModel  # type: ignore
except Exception:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import CostModel  # type: ignore


class LiveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Live Signal (Tk)")
        self.geometry("900x680")

        # Live runner state
        self._live_thread: Optional[threading.Thread] = None
        self._live_runner = None

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

        self.live_train_days = tk.IntVar(value=7)
        self.live_feat_minutes = tk.IntVar(value=2000)
        self.live_default_thresh = tk.DoubleVar(value=0.55)

        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label, width=28).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="we", padx=4, pady=3)

        cfg.columnconfigure(1, weight=1)
        add_row(cfg, 0, "Symbol (Binance spot):", ttk.Entry(cfg, textvariable=self.symbol_var))
        add_row(cfg, 1, "Taker fee per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.taker_var))
        add_row(cfg, 2, "Slippage per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.slip_var))
        add_row(cfg, 3, "Train days (history):", ttk.Spinbox(cfg, from_=2, to=120, textvariable=self.live_train_days))
        add_row(cfg, 4, "Feature minutes (recent):", ttk.Spinbox(cfg, from_=500, to=5000, increment=100, textvariable=self.live_feat_minutes))
        add_row(cfg, 5, "Default decision threshold:", ttk.Spinbox(cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.live_default_thresh))

        # Controls
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(8, 4))
        self.live_start = ttk.Button(btns, text="Start Live", command=self.start_live)
        self.live_start.pack(side=tk.LEFT)
        self.live_stop = ttk.Button(btns, text="Stop Live", command=self.stop_live, state=tk.DISABLED)
        self.live_stop.pack(side=tk.LEFT, padx=(8, 0))

        # Status
        live_stats = ttk.LabelFrame(frm, text="Live status", padding=10)
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

        # Logs
        live_logf = ttk.LabelFrame(frm, text="Live logs", padding=6)
        live_logf.pack(fill=tk.BOTH, expand=True)
        self.live_log = tk.Text(live_logf, wrap="word", height=24)
        self.live_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsb = ttk.Scrollbar(live_logf, orient="vertical", command=self.live_log.yview)
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.live_log.configure(yscrollcommand=lsb.set)

    # ---------------- Live actions ----------------
    def _append_live_log(self, text: str):
        self.live_log.insert(tk.END, text)
        self.live_log.see(tk.END)

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
            ts = msg.get("timestamp")
            prob = float(msg.get("prob_up")) if msg.get("prob_up") is not None else float("nan")
            conf = float(msg.get("confidence")) if msg.get("confidence") is not None else float("nan")
            sig = int(msg.get("signal")) if msg.get("signal") is not None else 0
            self.live_prob.set(f"{prob:.3f} (conf {conf:.2f}) @ {ts}")
            sig_str = "LONG" if sig == 1 else ("SHORT" if sig == -1 else "FLAT")
            self.live_signal.set(sig_str)
            self._append_live_log(f"Prediction {ts}: prob_up={prob:.3f}, confidence={conf:.2f}, signal={sig_str}\n")
        elif evt == "evaluation":
            ts = msg.get("timestamp")
            correct = bool(msg.get("correct"))
            self.live_eval.set(f"{'Correct' if correct else 'Wrong'} for {ts}")
            self._append_live_log(f"Evaluation for {ts}: {'Correct' if correct else 'Wrong'}\n")
        elif evt == "error":
            self._append_live_log(f"Error: {msg.get('message')}\n")
        else:
            self._append_live_log(str(msg) + "\n")

    def start_live(self):
        if self._live_thread and self._live_thread.is_alive():
            messagebox.showinfo("Live", "Live runner already active.")
            return
        self.live_status.set("Initializing...")

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
    app = LiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
