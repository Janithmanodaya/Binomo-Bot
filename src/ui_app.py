import os
import time
from typing import Dict

import pandas as pd
import streamlit as st
import plotly.express as px

from src.run_pipeline import CostModel, run_pipeline


st.set_page_config(page_title="Crypto Baseline Trainer", layout="wide")
st.title("Cost-aware 1m Crypto Direction — Trainer & Dashboard")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Symbol (Binance spot)", value="ETH/USDT")
    days = st.number_input("Lookback days", min_value=10, max_value=365, value=60, step=5)
    taker_fee_bps = st.number_input("Taker fee per side (bps)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
    slippage_bps = st.number_input("Slippage per side (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    folds = st.number_input("Walk-forward folds", min_value=1, max_value=20, value=5, step=1)
    val_days = st.number_input("Validation days per fold", min_value=1, max_value=60, value=10, step=1)
    prob_threshold = st.slider("Default decision threshold", min_value=0.50, max_value=0.80, value=0.55, step=0.01)

    st.markdown("---")
    run_btn = st.button("Run Training", type="primary")

progress_bar = st.progress(0)
status_text = st.empty()
fold_metrics_placeholder = st.empty()
results_placeholder = st.empty()

def on_progress(stage: str, p: float):
    status_text.text(f"{stage} ... ({int(p*100)}%)")
    progress_bar.progress(min(max(p, 0.0), 1.0))


def on_fold(fold: int, metrics: Dict[str, float]):
    with fold_metrics_placeholder.container():
        st.subheader(f"Fold {fold} metrics")
        cols = st.columns(6)
        cols[0].metric("AUC", f"{metrics['auc']:.3f}")
        cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[2].metric("Expectancy", f"{metrics['expectancy']:.3e}")
        cols[3].metric("Sharpe", f"{metrics['sharpe']:.2f}")
        cols[4].metric("Profit factor", f"{metrics['profit_factor']:.2f}")
        cols[5].metric("Trades", f"{metrics['trades']}")

if run_btn:
    try:
        cost = CostModel(taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps)
        results, summary, out_csv = run_pipeline(
            symbol=symbol,
            days=int(days),
            cost=cost,
            folds=int(folds),
            val_days=int(val_days),
            default_prob_threshold=float(prob_threshold),
            progress_callback=on_progress,
            fold_callback=on_fold,
        )
        progress_bar.progress(1.0)
        status_text.success(f"Training complete. Results saved to {out_csv}")

        with results_placeholder.container():
            st.subheader("Performance Report")

            # Cumulative PnL
            cum = results["pnl"].cumsum()
            fig = px.line(cum, title="Cumulative PnL (validation folds)", labels={"value": "PnL", "index": "Timestamp"})
            st.plotly_chart(fig, use_container_width=True)

            # Trades per minute
            trades_per_day = (results["signal"] != 0).resample("1D").sum()
            fig2 = px.bar(trades_per_day, title="Trades per day", labels={"value": "Trades", "index": "Day"})
            st.plotly_chart(fig2, use_container_width=True)

            # Summary table
            st.subheader("Per-fold summary")
            st.dataframe(summary.round(6), use_container_width=True)

            # Download links
            st.download_button(
                label="Download predictions.csv",
                data=open(out_csv, "rb").read(),
                file_name="predictions.csv",
                mime="text/csv",
            )

            # Show head of detailed results
            st.subheader("Sample of per-minute results")
            st.dataframe(results.head(200), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    # If predictions already exist, allow quick visualization without re-running
    default_path = "data/processed/predictions.csv"
    if os.path.exists(default_path):
        try:
            results = pd.read_csv(default_path, parse_dates=["timestamp"]).set_index("timestamp")
            st.info("Found existing results at data/processed/predictions.csv")
            cum = results["pnl"].cumsum()
            fig = px.line(cum, title="Cumulative PnL (existing results)", labels={"value": "PnL", "index": "Timestamp"})
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass