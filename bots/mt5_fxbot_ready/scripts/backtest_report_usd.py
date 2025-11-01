# scripts/backtest_report_usd.py
# Adds USD PnL, pips, lots, equity curve, PF, expectancy, DD
import numpy as np, pandas as pd
import MetaTrader5 as mt5
from mt5fx.utils import load_config
from mt5fx.backtest import run_backtest
from mt5fx.mt5_client import MT5

def pip_size(symbol: str) -> float:
    s = symbol.upper()
    return 0.01 if "JPY" in s else 0.0001

def main(symbol=None, timeframe=None, lookback=None):
    cfg = load_config("config.yaml")
    if symbol:   cfg["general"]["symbol"]    = symbol
    if timeframe:cfg["general"]["timeframe"] = timeframe
    if lookback: cfg["backtest"]["lookback"] = int(lookback)

    # run bar-level backtest (with BE/trailing/time-stop from your patched file)
    res = run_backtest(cfg)
    tr  = res["trades"].copy()
    print(res["summary"])
    if tr.empty:
        print("No trades"); return

    # figure out contract/tick info via MT5 client (accurate P&L in deposit currency)
    cli = MT5(); cli.init()
    info = cli.ensure_symbol(cfg["general"]["symbol"])
    resolved = getattr(cli, "last_symbol", cfg["general"]["symbol"])
    si = mt5.symbol_info(resolved)
    tick_size = float(si.trade_tick_size or si.point)
    tick_value= float(si.trade_tick_value or 0.0)
    lot      = cfg.get("backtest",{}).get("fixed_lots", cfg.get("risk",{}).get("fixed_lots", 0.10))
    mt5.shutdown()

    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    tr["month"] = tr["entry_time"].dt.to_period("M").astype(str)
    tr["lots"] = lot

    # pips & USD PnL
    ps = pip_size(resolved)
    sign = np.where(tr["side"].eq("long"), 1.0, -1.0)
    price_diff = sign * (tr["exit_price"] - tr["entry_price"])
    tr["pips"] = price_diff / ps
    # USD pnl using tick value/size * lots
    tr["pnl_usd"] = (price_diff / tick_size) * tick_value * tr["lots"]

    # equity curve
    tr["cum_pnl"] = tr["pnl_usd"].cumsum()
    roll_max = tr["cum_pnl"].cummax()
    tr["dd"] = tr["cum_pnl"] - roll_max
    max_dd = tr["dd"].min()

    # summary stats
    wins = tr["pnl_usd"] > 0
    pf = tr.loc[wins, "pnl_usd"].sum() / abs(tr.loc[~wins, "pnl_usd"].sum()) if (~wins).any() else float("inf")
    exp = tr["pnl_usd"].mean()

    print(f"\nDetails: lot={lot}, tick_value={tick_value}, tick_size={tick_size}, pip_size={ps}")
    print(f"Profit Factor: {pf:.2f} | Expectancy: ${exp:.2f} per trade | Max DD: ${max_dd:.2f}")

    print("\nBy month:")
    print(tr.groupby("month").agg(trades=("pnl_usd","count"),
                                  win_rate=("pnl_usd", lambda s: (s>0).mean()*100),
                                  net_usd=("pnl_usd","sum")).to_string())

    # show first rows
    print("\n=== Trades head ===")
    print(tr[["entry_time","exit_time","side","entry_price","exit_price","sl","tp","lots","pips","pnl_usd","result"]].head(12).to_string(index=False))

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    sym = args[0] if len(args)>0 else None
    tf  = args[1] if len(args)>1 else None
    lb  = args[2] if len(args)>2 else None
    main(sym, tf, lb)
