Baseline 1-minute crypto direction pipeline (cost-aware)

Overview
This repository contains a minimal, runnable baseline that:
- Downloads 1-minute OHLCV data from Binance (spot) via ccxt.
- Builds cost-aware, thresholded labels (UP / DOWN; ignores NEUTRAL).
- Engineers a compact set of robust features (returns, EMAs, RSI, ATR, vol stats).
- Trains a LightGBM classifier with walk-forward validation.
- Simulates a simple execution policy (market entry/exit next minute) with roundtrip costs (fees + slippage).
- Reports statistical and economic metrics (accuracy, AUC, expectancy, Sharpe) and saves a per-minute prediction/PnL file.
- NEW: Streamlit UI to configure runs, track progress, and view performance charts.
- NEW (Advanced): Live multi-level signal model with separate training, threshold tuning, and training progress bar.

Quick start
1) Install dependencies
   pip install -r requirements.txt

2) Run the baseline pipeline with defaults (ETH/USDT on Binance, ~60 days of data)
   python src/run_pipeline.py

Or use the UI (recommended for exploration)
   # Tkinter desktop UI (default when double-clicking run.py on Windows)
   python run.py
   # Explicitly launch Tkinter UI
   python run.py --tk

   # Streamlit web UI
   streamlit run src/ui_app.py
   # Or via helper
   python run.py --ui
   # If the browser doesn't open automatically, go to:
   # http://localhost:8501

UI tabs
- Backtest: run cost-aware walk-forward with progress and charts.
- Realtime (rich): train a rich-feature LightGBM model and preview minute-by-minute predictions with online correctness and confidence.
- Live (advanced): train a separate multi-level (5-class) model that generates signals with strengths {-2,-1,0,1,2}, with a training progress bar and a live preview that evaluates correctness and PnL online.

Common options (examples)
- Change symbol:
   python src/run_pipeline.py --symbol ETH/USDT

- Change lookback duration:
   python src/run_pipeline.py --days 120

- Set realistic costs (bps per side):
   python src/run_pipeline.py --taker-fee-bps 4 --slippage-bps 1

- Increase walk-forward folds and validation length:
   python src/run_pipeline.py --folds 6 --val-days 14

- Adjust decision threshold (trade only when p(up) is high/low enough):
   python src/run_pipeline.py --prob-threshold 0.55

Advanced live model
- Implementation: src/advanced_trainer.py
- Labels: multi-level using cost-aware thresholds tau = 2*(fee+slip)/10000
  - -2: next_ret < -2*tau
  - -1: [-2*tau, -tau)
  -  0: within [-tau, +tau]
  - +1: (tau, 2*tau]
  - +2: > 2*tau
- Model: LightGBM multiclass, class-weighted, rich multi-timeframe features.
- Threshold tuning: decision threshold chosen on validation to maximize realized PnL after costs using p_up = P(+1)+P(+2).
- Outputs: data/processed/advanced_model.txt and data/processed/advanced_meta.json
- Live preview saves: data/processed/advanced_preview.csv

Troubleshooting the UI (Windows)
- Activate your venv first:
   .\venv\Scripts\activate
- Ensure Streamlit is installed:
   pip show streamlit
- Launch specifically on a port (if 8501 is busy):
   streamlit run src/ui_app.py --server.port 8502
- If you see import errors for src.run_pipeline, run from the project root:
   cd path\to\project
   streamlit run src\ui_app.py

Outputs
- data/processed/predictions.csv: per-minute predictions, signals, and realized PnL for each fold.
- data/processed/realtime_model.txt / realtime_meta.json: realtime rich model.
- data/processed/advanced_model.txt / advanced_meta.json: advanced multi-level model.
- Console summary with statistical metrics (accuracy, AUC) and economic metrics (expectancy, Sharpe, max drawdown, profit factor).
- Streamlit dashboard with interactive charts and metrics.

Notes and limitations
- This is a simple market-in/market-out next-minute simulator; real execution can be improved with limit orders, partial fills, and depth-aware slippage models.
- Costs matter: the thresholded labels and decision thresholds are designed to reduce trading in noise.
- LightGBM is a strong tabular baseline; once you see stable positive expectancy in out-of-sample, consider adding order-book/trade-flow features and/or sequence models.

Safety
- The scripts are for research/backtesting. They do not place live orders.

License
MIT