import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px  # still used for any plots if needed later
import streamlit as st

# Ensure project root is on sys.path so `from src...` works when running as a script inside src/
try:
    from src.run_pipeline import CostModel  # type: ignore
    from src.realtime_pipeline import train_model as train_realtime_model  # type: ignore
    from src.advanced_trainer import train_multilevel_model, load_advanced_bundle, predict_latest  # type: ignore
except Exception:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.run_pipeline import CostModel  # type: ignore
    from src.realtime_pipeline import train_model as train_realtime_model  # type: ignore
    from src.advanced_trainer import train_multilevel_model, load_advanced_bundle, predict_latest  # type: ignore

from src.features_ta import build_rich_features  # type: ignore
import lightgbm as lgb  # type: ignore


st.set_page_config(page_title="Live Crypto Signal — Realtime + Advanced", layout="wide")
st.title("Live Crypto Signal — Realtime + Advanced (cost-aware)")

# Build/version banner to verify UI is up to date
UI_VERSION = "v0.4.0 (live-only UI)"
try:
    mtime = os.path.getmtime(__file__)
    ts = pd.to_datetime(mtime, unit="s")
    st.caption(f"UI build: {UI_VERSION} | ui_app.py last modified: {ts} UTC")
except Exception:
    st.caption(f"UI build: {UI_VERSION}")

tabs = st.tabs(["Realtime (rich)", "Live (advanced)"])

# ----------------------------- Realtime (rich) tab -----------------------------
with tabs[0]:
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
            rows: List[Dict] = []

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

                # Evaluate correctness and update live metrics/table
                if (ts in raw2.index) and (next_ts in raw2.index):
                    c0 = float(raw2.loc[ts, "close"])
                    c1 = float(raw2.loc[next_ts, "close"])
                    next_ret = float(np.log(c1) - np.log(c0))
                    tau = float(CostModel(taker_fee_bps=rt_fee, slippage_bps=rt_slip).roundtrip_cost_ret)

                    if signal == 1:
                        correct = bool(next_ret > tau)
                        pnl = float(next_ret - tau)
                    elif signal == -1:
                        correct = bool(next_ret < -tau)
                        pnl = float(-next_ret - tau)
                    else:
                        correct = bool(abs(next_ret) <= tau)
                        pnl = 0.0

                    eligible = (signal != 0) and (confidence >= min_conf)

                    # Update last row with evaluation
                    rows[-1]["next_ret"] = next_ret
                    rows[-1]["correct"] = correct
                    rows[-1]["eligible"] = bool(eligible)
                    rows[-1]["pnl"] = pnl

                    # Build DataFrame and compute summary metrics
                    df_rows = pd.DataFrame(rows)
                    if "eligible" in df_rows.columns:
                        df_elig = df_rows[df_rows["eligible"] == True]
                    else:
                        df_elig = pd.DataFrame(columns=df_rows.columns)

                    trades = int((df_rows["signal"] != 0).sum()) if "signal" in df_rows.columns else 0
                    elig_trades = int(len(df_elig))
                    wins = int(df_elig["correct"].sum()) if ("correct" in df_elig.columns and elig_trades > 0) else 0
                    win_rate = (wins / elig_trades) if elig_trades > 0 else 0.0
                    cum_pnl = float(df_elig["pnl"].sum()) if ("pnl" in df_elig.columns and elig_trades > 0) else 0.0

                    with ph_metrics.container():
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Eligible trades", f"{elig_trades}")
                        mc2.metric("Win rate (eligible)", f"{win_rate * 100:.1f}%")
                        mc3.metric("Cum PnL (eligible)", f"{cum_pnl:.3e}")
                        mc4.metric("All signals", f"{trades}")

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

# ----------------------------- Advanced Live tab -----------------------------
with tabs[1]:
    st.subheader("Live (advanced): multi-level signals with training progress")
    colA, colB, colC = st.columns(3)
    adv_symbol = colA.text_input("Symbol", value="ETH/USDT", key="adv_symbol")
    adv_days = colB.number_input("Train lookback (days)", min_value=15, max_value=365, value=120, step=5, key="adv_days")
    adv_minutes = colC.number_input("Preview minutes", min_value=1, max_value=240, value=10, step=1, key="adv_preview")

    colD, colE, colF = st.columns(3)
    adv_fee = colD.number_input("Taker fee per side (bps)", min_value=0.0, max_value=50.0, value=4.0, step=0.5, key="adv_fee")
    adv_slip = colE.number_input("Slippage per side (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="adv_slip")
    adv_min_conf = float(colF.slider("Min confidence to count a trade", min_value=0.00, max_value=1.00, value=0.40, step=0.01, key="adv_min_conf"))

    train_adv_btn = st.button("Train/Refresh advanced model", key="adv_train")
    run_adv_btn = st.button("Run advanced live preview", type="primary", key="adv_run")

    adv_progress = st.progress(0.0)
    adv_status = st.empty()

    if train_adv_btn:
        try:
            cost = CostModel(taker_fee_bps=adv_fee, slippage_bps=adv_slip)
            def on_prog(stage: str, p: float):
                adv_status.text(f"{stage} ... ({int(p*100)}%)")
                adv_progress.progress(min(max(p, 0.0), 1.0))
            model_path, meta_path = train_multilevel_model(adv_symbol, int(adv_days), cost, progress=on_prog)
            adv_progress.progress(1.0)
            adv_status.success(f"Advanced model trained.\nModel: {model_path}\nMeta: {meta_path}")
        except Exception as e:
            st.error(f"Advanced training error: {e}")

    if run_adv_btn:
        try:
            bundle = load_advanced_bundle()
            if bundle is None:
                st.warning("No advanced model found. Training one now...")
                cost = CostModel(taker_fee_bps=adv_fee, slippage_bps=adv_slip)
                def on_prog(stage: str, p: float):
                    adv_status.text(f"{stage} ... ({int(p*100)}%)")
                    adv_progress.progress(min(max(p, 0.0), 1.0))
                train_multilevel_model(adv_symbol, int(adv_days), cost, progress=on_prog)
                bundle = load_advanced_bundle()

            assert bundle is not None
            ph_status = st.empty()
            ph_metrics = st.empty()
            ph_table = st.empty()
            rows: List[Dict] = []

            for i in range(int(adv_minutes)):
                ts, prob_up, signal, confidence = predict_latest(adv_symbol, feature_minutes=1200, bundle=bundle)
                signal_str = {2: "LONG x2", 1: "LONG", 0: "FLAT", -1: "SHORT", -2: "SHORT x2"}[int(signal)]

                rows.append(dict(timestamp=ts, prob_up=prob_up, confidence=confidence, signal=signal, signal_str=signal_str))
                df_rows = pd.DataFrame(rows)
                ph_table.dataframe(df_rows.tail(50), use_container_width=True)
                ph_status.info(f"[{i+1}/{int(adv_minutes)}] {ts} prob_up={prob_up:.4f} conf={confidence:.3f} signal={signal_str}")

                # Wait for next bar and evaluate correctness vs cost-aware threshold
                next_ts = (pd.Timestamp(ts).floor("T") + pd.Timedelta(minutes=1)).tz_convert("UTC")
                sleep_s = max(2.0, (next_ts - pd.Timestamp.now(tz="UTC")).total_seconds() + 2.0)
                time.sleep(sleep_s)

                # Evaluate
                import ccxt
                ex = ccxt.binance({"enableRateLimit": True})
                raw2_rows = ex.fetch_ohlcv(adv_symbol, timeframe="1m", limit=5)
                raw2 = pd.DataFrame(raw2_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                raw2["timestamp"] = pd.to_datetime(raw2["timestamp"], unit="ms", utc=True)
                raw2 = raw2.set_index("timestamp").sort_index().drop_duplicates()
                if (ts in raw2.index) and (next_ts in raw2.index):
                    c0 = float(raw2.loc[ts, "close"])
                    c1 = float(raw2.loc[next_ts, "close"])
                    next_ret = float(np.log(c1) - np.log(c0))
                    meta = bundle["meta"]
                    tau = 2.0 * (float(meta["taker_fee_bps"]) + float(meta["slippage_bps"])) / 10000.0
                    # For strength 2, require > 2*tau for correctness; for strength 1, > tau
                    if signal > 0:
                        req = (2 * tau) if signal == 2 else tau
                        correct = bool(next_ret > req)
                        pnl = float(next_ret - req)
                    elif signal < 0:
                        req = (2 * tau) if signal == -2 else tau
                        correct = bool(next_ret < -req)
                        pnl = float(-next_ret - req)
                    else:
                        correct = bool(abs(next_ret) <= tau)
                        pnl = 0.0

                    eligible = (signal != 0) and (confidence >= adv_min_conf)
                    rows[-1]["next_ret"] = next_ret
                    rows[-1]["correct"] = correct
                    rows[-1]["eligible"] = bool(eligible)
                    rows[-1]["pnl"] = pnl

                    df_rows = pd.DataFrame(rows)
                    if "eligible" in df_rows.columns:
                        df_elig = df_rows[df_rows["eligible"] == True]
                    else:
                        df_elig = pd.DataFrame(columns=df_rows.columns)

                    trades = int((df_rows["signal"] != 0).sum())
                    elig_trades = int(len(df_elig))
                    wins = int(df_elig["correct"].sum()) if elig_trades > 0 else 0
                    win_rate = (wins / elig_trades) if elig_trades > 0 else 0.0
                    cum_pnl = float(df_elig["pnl"].sum()) if ("pnl" in df_elig.columns and elig_trades > 0) else 0.0

                    with ph_metrics.container():
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Eligible trades", f"{elig_trades}")
                        mc2.metric("Win rate (eligible)", f"{win_rate * 100:.1f}%")
                        mc3.metric("Cum PnL (eligible)", f"{cum_pnl:.3e}")
                        mc4.metric("All signals", f"{trades}")

                    ph_table.dataframe(df_rows.tail(50), use_container_width=True)

            # Save preview to CSV
            os.makedirs("data/processed", exist_ok=True)
            out_csv = "data/processed/advanced_preview.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            st.success(f"Advanced realtime preview saved: {out_csv}")
            st.download_button(
                "Download advanced_preview.csv",
                data=open(out_csv, "rb").read(),
                file_name="advanced_preview.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Advanced preview error: {e}")