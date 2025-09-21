import argparse
import json
import threading
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
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.model_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    threshold = float(args.threshold if args.threshold is not None else meta.get("tuned_threshold", 0.55))
    cost = CostModel(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)

    stop_flag = {"stop": False}
    def should_stop():
        return stop_flag["stop"]

    print(f"Starting real-time predictor on {args.symbol} with threshold={threshold:.2f}")
    def on_tick(ev: Dict):
        if "msg" in ev:
            print(ev["msg"])
        if "prob_up" in ev:
            print(f"prob_up={ev['prob_up']:.4f}, confidence={ev.get('confidence', 0.0):.3f}, signal={ev['signal']}")
        if "correct" in ev:
            print(f"Evaluation: next_ret={ev['next_ret']:.6e}, correct={ev['correct']}")

    try:
        continuous_predict(
            symbol=args.symbol,
            model_path=args.model_file,
            feature_names=feature_names,
            threshold=threshold,
            cost=cost,
            on_tick=on_tick,
            stop_flag=should_stop,
        )
    except KeyboardInterrupt:
        stop_flag["stop"] = True
        print("Stopping...")

if __name__ == "__main__":
    main()