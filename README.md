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

Quick start
1) Install dependencies
   pip install -r requirements.txt

2) Run the pipeline with defaults (ETH/USDT on Binance, ~60 days of data)
   python src/run_pipeline.py

Or use the UI (recommended for exploration)
   # Option A (direct)
   streamlit run src/ui_app.py
   # Option B (via helper)
   python run.py --ui
   # If the browser doesn't open automatically, go to:
   # http://localhost:8501

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

What this baseline does
- Fetches historical 1m OHLCV (UTC) and stores a cleaned parquet in data/raw/.
- Computes features at time t using only information available up to t (no lookahead).
- Creates labels using a return threshold tau that approximates roundtrip costs:
    tau = 2 * (taker-fee-bps + slippage-bps) / 10000
  Label UP if next_return > +tau, DOWN if next_return < -tau, else NEUTRAL (dropped from training).
- Trains LightGBM on expanding windows, validates on out-of-sample rolling segments, and tunes the decision probability threshold on each validation fold to maximize realized PnL after costs.

UI features
- Configure dataset size (days), fees/slippage, folds, validation horizon, and threshold.
- Live progress bar and fold-by-fold metrics as training proceeds.
- Charts: cumulative PnL over time, trades per day.
- Per-fold summary table and download button for predictions.csv.
- If predictions already exist, the UI can visualize them without rerunning.

Troubleshooting the UI (Windows)
- Activate your venv first:
   .\\venv\\Scripts\\activate
- Ensure Streamlit is installed:
   pip show streamlit
- Launch specifically on a port (if 8501 is busy):
   streamlit run src/ui_app.py --server.port 8502
- If you see import errors for src.run_pipeline, run from the project root:
   cd path\\to\\project
   streamlit run src\\ui_app.py

Outputs
- data/processed/predictions.csv: per-minute predictions, signals, and realized PnL for each fold.
- Console summary with statistical metrics (accuracy, AUC) and economic metrics (expectancy, Sharpe, max drawdown, profit factor).
- Streamlit dashboard with interactive charts and metrics.

Notes and limitations
- This is a simple market-in/market-out next-minute simulator; real execution can be improved with limit orders, partial fills, and depth-aware slippage models.
- Costs matter: the thresholded labels and decision thresholds are designed to reduce trading in noise.
- LightGBM is a strong tabular baseline; once you see stable positive expectancy in out-of-sample, consider adding order-book/trade-flow features and/or sequence models.

Safety
- The script is for research/backtesting. It does not place live orders.

License
MIT