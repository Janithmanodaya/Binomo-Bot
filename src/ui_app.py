import os
import time
from typing import Dict, List, Optional
import numpy as np

import pandas as pd
import streamlit as st
import plotly.express as px

# Ensure project root is on sys.path so `from src...` works when running as a script inside src/
try:
    from src.run_pipeline import CostModel, run_pipeline  # type: ignore
    from src.realtime_pipeline import train_model as train_realtime_model  # type: ignore
except Exception:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import CostModel, run_pipeline  # type: ignore
    from src.realtime_pipeline import train_model as train_realtime_model  # type: ignore

from src.features_ta import build_rich_features  # type: ignore
import lightgbm as lgb  # type: ignore


st.set_page_config(page_title="Crypto Baseline Trainer", layout="wide")
st.title("Cost-aware 1m Crypto Direction — Trainer & Dashboard")

tabs = st.tabs(["Backtest", "Realtime (rich)"])

# ----------------------------- Backtest tab -----------------------------
with tabs[0]:
    with st.sidebar:
        st.header("Backtest configuration")
        symbol = st.text_input("Symbol (Binance spot)", value="ETH/USDT", key="bt_symbol")
        days = st.number_input("Lookback days", min_value=10, max_value=365, value=60, step=5, key="bt_days")
        taker_fee_bps = st.number_input("Taker fee per side (bps)", min_value=0.0, max_value=50.0, value=4.0, step=0.5, key="bt_fee")
        slippage_bps = st.number_input("Slippage per side (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="bt_slip")
        folds = st.number_input("Walk-forward folds", min_value=1, max_value=20, value=5, step=1, key="bt_folds")
        val_days = st.number_input("Validation days per fold", min_value=1, max_value=60, value=10, step=1, key="bt_valdays")
        prob_threshold = st.slider("Default decision threshold", min_value=0.50, max_value=0.80, value=0.55, step=0.01, key="bt_thresh")

        st.markdown("---")
        run_btn = st.button("Run Training", type="primary", key="bt_run")

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

# ----------------------------- Realtime tab -----------------------------
with tabs[1]:
    st.subheader("Realtime predictions with rich multi-timeframe features")
    colA, colB, colC = st.columns(3)
    rt_symbol = colA.text_input("Symbol", value="ETH/USDT", key="rt_symbol")
    rt_days = colB.number_input("Train lookback (days)", min_value=10, max_value=365, value=90, step=5, key="rt_days")
    rt_minutes = colC.number_input("Preview minutes", min_value=1, max_value=240, value=10, step=1, key="rt_preview")

    colD, colE, colF = st.columns(3)
    rt_fee = colD.number_input("Taker fee per side (bps)", min_value=0.0, max_value=50.0, value=4.0, step=0.5, key="rt_fee")
    rt_slip = colE.number_input("Slippage per side (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="rt_slip")
    min_conf = float(colF.slider("Min confidence to count a trade", min_value=0.00, max_value=1.00, value=0.30, step=0.01, key="rt_min_conf"))

    train_btn = st.button("Train/Refresh realtime model", key="rt_train")
    run_preview_btn = st.button("Run realtime preview", type="primary", key="rt_run")

    model_path = "data/processed/realtime_model.txt"
    meta_path = "data/processed/realtime_meta.json"

    def load_realtime_model() -> Optional[Dict]:
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            return None
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        booster = lgb.Booster(model_file=model_path)
        return {"booster": booster, "meta": meta}

    if train_btn:
        try:
            cost = CostModel(taker_fee_bps=rt_fee, slippage_bps=rt_slip)
            model_path_out, meta_path_out = train_realtime_model(rt_symbol, int(rt_days), cost)
            st.success(f"Trained realtime model.\nModel: {model_path_out}\nMeta: {meta_path_out}")
        except Exception as e:
            st.error(f"Realtime training error: {e}")

    # Preview loop
    if run_preview_btn:
        try:
            cost = CostModel(taker_fee_bps=rt_fee, slippage_bps=rt_slip)
            bundle = load_realtime_model()
            if bundle is None:
                st.warning("No realtime model found. Training one now...")
                model_path_out, meta_path_out = train_realtime_model(rt_symbol, int(rt_days), cost)
                bundle = load_realtime_model()

            assert bundle is not None
            booster: lgb.Booster = bundle["booster"]
            meta = bundle["meta"]
            feature_names: List[str] = list(meta.get("feature_names", []))
            threshold: float = float(meta.get("threshold", 0.55))

            st.info(f"Using threshold={threshold:.2f}")
            ph_status = st.empty()
            ph_metrics = st.empty()
            ph_table = st.empty()
            rows = []

            for i in range(int(rt_minutes)):
                # Fetch recent candles (limit 800) without 'since' to get the freshest bars
                import ccxt
                ex = ccxt.binance({"enableRateLimit": True})
                raw_rows = ex.fetch_ohlcv(rt_symbol, timeframe="1m", limit=800)
                if not raw_rows:
                    raise RuntimeError("No OHLCV received.")
                raw = pd.DataFrame(raw_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
                raw = raw.set_index("timestamp").sort_index().drop_duplicates()

                feats = build_rich_features(raw)
                if feats.empty:
                    ph_status.warning("Insufficient features; waiting for next minute...")
                    time.sleep(5)
                    continue

                ts = feats.index[-1]
                X_row = feats.iloc[[-1]].copy()
                # Align columns
                for col in feature_names:
                    if col not in X_row.columns:
                        X_row[col] = 0.0
                X_row = X_row[feature_names] if feature_names else X_row

                prob_up = float(booster.predict(X_row)[0])
                # Decision-aware confidence in [0,1]
                lo = 1.0 - threshold
                if prob_up >= threshold:
                    confidence = float(max(0.0, min(1.0, (prob_up - threshold) / (1.0 - threshold))))
                elif prob_up <= lo:
                    confidence = float(max(0.0, min(1.0, (lo - prob_up) / (1.0 - threshold))))
                else:
                    confidence = 0.0
                signal = int(1 if prob_up > threshold else (-1 if prob_up < 1 - threshold else 0))

                # Track row; eligibility will be set after evaluation
                rows.append(dict(timestamp=ts, prob_up=prob_up, confidence=confidence, signal=signal))
                df_rows = pd.DataFrame(rows)
                ph_table.dataframe(df_rows.tail(50), use_container_width=True)

                ph_status.info(f"[{i+1}/{int(rt_minutes)}] {ts} prob_up={prob_up:.4f} conf={confidence:.3f} signal={signal}")

                # Wait until next minute ends, then evaluate correctness
                next_ts = (pd.Timestamp(ts).floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")
                sleep_s = max(2.0, (next_ts - pd.Timestamp.now(tz="UTC")).total_seconds() + 2.0)
                time.sleep(sleep_s)

                # Refresh small window to evaluate
                raw2_rows = ex.fetch_ohlcv(rt_symbol, timeframe="1m", limit=5)
                raw2 = pd.DataFrame(raw2_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                raw2["timestamp"] = pd.to_datetime(raw2["timestamp"], unit="ms", utc=True)
                raw2 = raw2.set_index("timestamp").sort_index().drop_duplicates()

                correct = None
                if ts in raw2.index and next_ts in raw2.index:
                    c0 = float(raw2.loc[ts, "close"])
                    c1 = float(raw2.loc[next_ts, "close"])
                    next_ret = float(np.log(c1) - np.log(c0))  # type: ignore[name-defined]
                    tau = cost.roundtrip_cost_ret
                    if signal == 1:
                        correct = next_ret > tau
                        pnl = next_ret - tau
                    elif signal == -1:
                        correct = next_ret < -tau
                        pnl = -next_ret - tau
                    else:
                        correct = abs(next_ret) <= tau
                        pnl = 0.0

                    # mark eligibility by confidence threshold and non-flat signal
                    eligible = bool((signal != 0) and (confidence >= min_conf))
                    rows[-1]["next_ret"] = next_ret
                    rows[-1]["correct"] = bool(correct)
                    rows[-1]["eligible"] = eligible
                    rows[-1]["pnl"] = float(pnl)

                    # Update metrics on eligible trades
                    df_rows = pd.DataFrame(rows)
                    df_elig = df_rows[(df_rows.get("eligible", False) == True)]
                    total = int(len(df_rows))
                    trades = int((df_rows["signal"] != 0).sum()) if "signal" in df_rows else 0
                    elig_trades = int(len(df_elig))
                    wins = int(df_elig["correct"].sum()) if "correct" in df_elig else 0
                    win_rate = (wins / max(elig_trades, 1)) if elig_trades > 0 else 0.0
                    cum_pnl = float(df_elig.get("pnl", pd.Series(dtype=float)).sum()) if elig_trades > 0 else 0.0

                    with ph_metrics.container():
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Eligible trades", f"{elig_trades}")
                        mc2.metric("Win rate (eligible)", f"{win_rate*100:.1f}%")
                        mc3.metric("Cum PnL (eligible)", f"{cum_pnl:.3e}")
                        mc4.metric("All signals", f"{trades}")

                    # Show latest table (tail)
                    ph_table.dataframe(df_rows.tail(50), use_container_width=True)

            # Save preview to CSV
            os.makedirs("data/processed", exist_ok=True)
            out_csv = "data/processed/realtime_preview.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            st.success(f"Realtime preview saved: {out_csv}")
            st.download_button(
                "Download realtime_preview.csv",
                data=open(out_csv, "rb").read(),
                file_name="realtime_preview.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Realtime preview error: {e}")