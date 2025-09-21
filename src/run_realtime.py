import argparse
import json
from typing import Dict

from src.run_pipeline import CostModel
from src.realtime_predictor import continuous_predict


def parse_args():
    p = argparse.ArgumentParser(description="Continuous real-time predictor (per-minute)")
    p.add_argument("--symbol", type=str, default="ETH/USDT", help="Binance spot symbol")
    p.add_argument("--model-meta", type=str, default="data/processed/rich_meta.json", help="Path to meta JSON")
    p.add_argument("--model-file", type=str, default="data/processed/rich_model.txt", help="Path to LightGBM model")
    p.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: from meta)")
    p.add_argument("--taker-fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--min-confidence", type=float, default=0.0, help="Only count trades with confidence >= this")
    p.add_argument("--report-csv", type=str, default="data/processed/live_report.csv", help="Path to write live report CSV")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.model_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    threshold = float(args.threshold if args.threshold is not None else meta.get("tuned_threshold", 0.55))
    cost = CostModel(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)

    print(f"Starting real-time predictor on {args.symbol} with threshold={threshold:.2f} (min_conf={args.min_confidence:.2f})")

    def on_tick(ev: Dict):
        if "msg" in ev:
            print(ev["msg"])
        if "prob_up" in ev:
            print(f"prob_up={ev['prob_up']:.4f}, confidence={ev.get('confidence', 0.0):.3f}, signal={ev['signal']}")
        if "correct" in ev:
            parts = [f"Evaluation: next_ret={ev['next_ret']:.6e}, correct={ev['correct']}"]
            if "eligible" in ev:
                parts.append(f"eligible={ev['eligible']}")
            if "metrics" in ev:
                m = ev["metrics"]
                parts.append(f"eligible_trades={m.get('eligible_trades', 0)}")
                parts.append(f"win_rate={m.get('win_rate', 0.0):.3f}")
                parts.append(f"cum_pnl={m.get('cum_pnl', 0.0):.3e}")
            print(", ".join(parts))

    try:
        continuous_predict(
            symbol=args.symbol,
            model_path=args.model_file,
            feature_names=feature_names,
            threshold=threshold,
            cost=cost,
            on_tick=on_tick,
            stop_flag=None,
            min_confidence=float(args.min_confidence),
            report_path=args.report_csv,
        )
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()