import os
import sys
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, Callable

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Ensure project root is on sys.path so `from src...` works regardless of CWD
try:
    from src.run_pipeline import CostModel, run_pipeline  # type: ignore
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import CostModel, run_pipeline  # type: ignore


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
        self.geometry("900x700")

        self._stdout_queue: "queue.Queue[str]" = queue.Queue()
        self._stdout_prev: Optional[object] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._build_ui()
        self.after(100, self._poll_stdout)

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Config inputs
        cfg = ttk.LabelFrame(frm, text="Configuration", padding=10)
        cfg.pack(fill=tk.X)

        self.symbol_var = tk.StringVar(value="ETH/USDT")
        self.days_var = tk.IntVar(value=60)
        self.taker_var = tk.DoubleVar(value=4.0)
        self.slip_var = tk.DoubleVar(value=1.0)
        self.folds_var = tk.IntVar(value=5)
        self.valdays_var = tk.IntVar(value=10)
        self.threshold_var = tk.DoubleVar(value=0.55)

        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label, width=24).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            widget.grid(row=r, column=1, sticky="we", padx=4, pady=3)

        cfg.columnconfigure(1, weight=1)
        add_row(cfg, 0, "Symbol (Binance spot):", ttk.Entry(cfg, textvariable=self.symbol_var))
        add_row(cfg, 1, "Lookback days:", ttk.Spinbox(cfg, from_=10, to=365, textvariable=self.days_var, increment=5))
        add_row(cfg, 2, "Taker fee per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.taker_var))
        add_row(cfg, 3, "Slippage per side (bps):", ttk.Spinbox(cfg, from_=0.0, to=50.0, increment=0.5, textvariable=self.slip_var))
        add_row(cfg, 4, "Walk-forward folds:", ttk.Spinbox(cfg, from_=1, to=20, textvariable=self.folds_var))
        add_row(cfg, 5, "Validation days per fold:", ttk.Spinbox(cfg, from_=1, to=60, textvariable=self.valdays_var))
        add_row(cfg, 6, "Default decision threshold:", ttk.Spinbox(cfg, from_=0.50, to=0.80, increment=0.01, textvariable=self.threshold_var))

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(8, 4))
        self.start_btn = ttk.Button(btns, text="Run Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.open_out_btn = ttk.Button(btns, text="Open predictions.csv", command=self.open_output, state=tk.DISABLED)
        self.open_out_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Progress
        prog = ttk.Frame(frm)
        prog.pack(fill=tk.X, pady=(4, 8))
        self.status_var = tk.StringVar(value="Idle")
        self.status_lbl = ttk.Label(prog, textvariable=self.status_var)
        self.status_lbl.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, expand=True, padx=(8, 0))

        # Log output
        logf = ttk.LabelFrame(frm, text="Logs", padding=6)
        logf.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(logf, wrap="word", height=25)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=sb.set)

        # Footer
        self.footer = ttk.Label(frm, text="Predictions: (not generated yet)")
        self.footer.pack(fill=tk.X, pady=(6, 0))

    def _append_log(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _poll_stdout(self):
        try:
            while True:
                s = self._stdout_queue.get_nowait()
                self._append_log(s)
        except queue.Empty:
            pass
        self.after(100, self._poll_stdout)

    def start_training(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Training", "Training is already running.")
            return

        # Reset UI
        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self.open_out_btn.configure(state=tk.DISABLED)
        self._stop_flag.clear()
        self.footer.configure(text="Predictions: (running)")

        # Redirect stdout to log
        self._stdout_prev = sys.stdout
        sys.stdout = StdoutRedirector(self._stdout_queue)  # type: ignore

        # Start worker thread
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
            # Restore stdout
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


def main():
    app = TrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()