#!/usr/bin/env python3
import argparse
import json
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

    p_wf = sub.add_parser("walk-forward", help="Run walk-forward validation")
    p_wf.add_argument("--symbol", required=False, help="e.g., EURUSD")
    p_wf.add_argument("--timeframe", required=False, help="e.g., M15")
    p_wf.add_argument("--lookback", type=int, required=False, help="Total candles to fetch")
    p_wf.add_argument("--train-bars", type=int, default=4000, help="Training window size")
    p_wf.add_argument("--test-bars", type=int, default=1000, help="Test window size")
    p_wf.add_argument("--step-bars", type=int, default=1000, help="Step size between windows")

    p_journal = sub.add_parser("journal", help="Show trade journal stats")

    args = parser.parse_args()
    cfg = load_config("config.yaml")

    if hasattr(args, "symbol") and args.symbol:
        cfg["general"]["symbol"] = args.symbol
    if hasattr(args, "timeframe") and args.timeframe:
        cfg["general"]["timeframe"] = args.timeframe
    if args.cmd in ("backtest", "walk-forward") and hasattr(args, "lookback") and args.lookback:
        cfg["backtest"]["lookback"] = args.lookback

    ensure_dirs(cfg["general"]["out_dir"])
    log = setup_logging(cfg["general"]["out_dir"])

    if args.cmd == "backtest":
        log.info("Starting backtest...")
        res = run_backtest(cfg)
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        for k, v in res["summary"].items():
            print(f"  {k:>20s}: {v}")
        print("=" * 60)
        print("\n=== Recent Trades ===")
        print(res["trades"].tail(10).to_string(index=False))

    elif args.cmd == "live":
        log.info("Starting LIVE engine... (DEMO!)")
        engine = LiveEngine(cfg)
        engine.run()

    elif args.cmd == "walk-forward":
        log.info("Starting walk-forward validation...")
        from mt5fx.walk_forward import walk_forward_validation
        from mt5fx.data import load_rates

        g = cfg["general"]
        b = cfg["backtest"]
        df = load_rates(g["symbol"], g["timeframe"], count=b["lookback"])

        result = walk_forward_validation(
            cfg, df,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
        )

        print("\n" + "=" * 60)
        print("  WALK-FORWARD VALIDATION")
        print("=" * 60)
        for w in result["windows"]:
            print(f"  Window {w['window']}: IS WR={w['is_win_rate']}% PF={w['is_pf']} | OOS WR={w['oos_win_rate']}% PF={w['oos_pf']} PnL=${w['oos_total_pnl']}")
        print("-" * 60)
        if result["oos_summary"]:
            print("  OUT-OF-SAMPLE AGGREGATE:")
            for k, v in result["oos_summary"].items():
                print(f"    {k:>25s}: {v}")
        print("=" * 60)

    elif args.cmd == "journal":
        from mt5fx.journal import TradeJournal
        import os
        db_path = os.path.join(cfg["general"]["out_dir"], "trade_journal.db")
        journal = TradeJournal(db_path)
        stats = journal.get_stats()
        print("\n" + "=" * 60)
        print("  TRADE JOURNAL STATS")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k:>20s}: {v}")
        print("=" * 60)
        recent = journal.get_recent_trades(10)
        if recent:
            print("\n=== Last 10 Trades ===")
            for t in recent:
                print(f"  #{t['ticket']} {t['symbol']} {t['side']} entry={t['entry_price']} pnl={t.get('pnl_usd', 'open')} {t.get('result', 'open')}")
        journal.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
