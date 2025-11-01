#!/usr/bin/env python3
import argparse, json
from mt5fx.logging_utils import setup_logging
from mt5fx.engine import LiveEngine
from mt5fx.backtest import run_backtest
from mt5fx.utils import load_config, ensure_dirs

def main():
    parser = argparse.ArgumentParser(description="MT5 FXBot")
    sub = parser.add_subparsers(dest="cmd")

    p_back = sub.add_parser("backtest", help="Run backtest")
    p_back.add_argument("--symbol", required=False, help="e.g., EURUSD")
    p_back.add_argument("--timeframe", required=False, help="e.g., M5")
    p_back.add_argument("--lookback", type=int, required=False, help="Candles to fetch")

    p_live = sub.add_parser("live", help="Run live trading (DEMO first!)")
    p_live.add_argument("--symbol", required=False, help="e.g., EURUSD")
    p_live.add_argument("--timeframe", required=False, help="e.g., M5")

    args = parser.parse_args()
    cfg = load_config("config.yaml")

    if args.symbol: cfg["general"]["symbol"] = args.symbol
    if args.timeframe: cfg["general"]["timeframe"] = args.timeframe
    if args.cmd == "backtest" and args.lookback:
        cfg["backtest"]["lookback"] = args.lookback

    ensure_dirs(cfg["general"]["out_dir"])
    log = setup_logging(cfg["general"]["out_dir"])

    if args.cmd == "backtest":
        log.info("Starting backtest...")
        res = run_backtest(cfg)
        print(json.dumps(res["summary"], indent=2))
        print("\n=== Trades head ===")
        print(res["trades"].head(10).to_string(index=False))
    elif args.cmd == "live":
        log.info("Starting LIVE engine... (DEMO!)")
        engine = LiveEngine(cfg)
        engine.run()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
