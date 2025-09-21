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