# scripts/pretty_report.py
# Backtest & save CSVs (trades + equity), then print a concise console report.

from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
# allow running this file directly: python scripts/pretty_report.py ...
if __name__ == "__main__" and __package__ is None:
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# --- strategy loader (same contract as your engine uses) ---
from mt5fx.strategy_loader import compute_plugin_signal
from mt5fx.mt5_client import MT5
from mt5fx.strategy import attach_atr   # reuses your ATR helper

def _timeframe(tf_str: str):
    tf_str = tf_str.upper()
    m = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
    }
    return m[tf_str]

def _iso(ts):
    # safe ISO text so Excel shows text instead of auto-format
    if isinstance(ts, (int, float, np.integer, np.floating)):
        ts = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)

def _pips_value(symbol: str) -> float:
    """
    Rough pip value per 1.0 lot for USD-quoted majors (EURUSD, GBPUSD...).
    We’ll use 10 USD per pip -> 10000 USD per 1.0000 price unit.
    This matches the quick calc you were already using in prior code.
    """
    return 10000.0

def _compute_equity(trades: pd.DataFrame, start_equity: float, symbol: str) -> pd.Series:
    pv = _pips_value(symbol)
    pnl = np.where(
        trades["side"].values == "long",
        (trades["exit_price"].values - trades["entry_price"].values) * pv,
        (trades["entry_price"].values - trades["exit_price"].values) * pv,
    )
    eq = np.cumsum(np.insert(pnl, 0, start_equity))  # first point = start equity
    # drop the first inserted value to align with trades rows while still returning a full equity curve separately
    trades["pnl"] = pnl
    trades["equity"] = start_equity + np.cumsum(pnl)
    return pd.Series(eq, index=[-1] + list(trades.index))  # index -1 = start equity

def backtest(symbol: str,
             timeframe: str,
             lookback: int,
             plugin_module: str,
             plugin_function: str = "generate_signals",
             start_equity: float = 10000.0) -> tuple[pd.DataFrame, pd.Series, dict]:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available.")
    # connect
    MT5().init()
    tf = _timeframe(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, lookback)
    if rates is None or len(rates) == 0:
        raise RuntimeError("No rates returned from MT5. Is the symbol/timeframe visible?")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"open":"bid_o","high":"bid_h","low":"bid_l","close":"bid_c"}, inplace=True)

    # fabricate ask/mid from spread like your engine does
    si = mt5.symbol_info(symbol)
    spread_px = (df["spread"].astype(float) * float(si.point)) if "spread" in df.columns else float(si.spread) * float(si.point)
    for k in ("o","h","l","c"):
        df[f"ask_{k}"] = df[f"bid_{k}"] + spread_px
        df[f"mid_{k}"] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2.0

    # signal on **closed bars**, execute next bar open
    sig = compute_plugin_signal(
        df,
        plugin_module=plugin_module,
        func_name=plugin_function,
        shift_next_bar=True,
    )
    df = attach_atr(df, n=14)  # ATR is only needed for stop/tp emulation in your plugin exits

    # simulate simple mean-reversion execution:
    # entry at next bar open (ask for long, bid for short); TP/SL result already encoded by your framework backtester,
    # but here we reconstruct based on "mean TP first" + ATR SL/TP multipliers from the framework backtester.
    # To keep behaviour consistent with your last good backtest, we use exactly TP/SL from your trades stream below.

    # Build trades from signal turns
    trades = []
    pos = 0
    for i in range(len(df)-1):  # next bar exists
        s = int(sig.iloc[i])
        if pos == 0 and s != 0:
            side = "long" if s > 0 else "short"
            entry_time = df.iloc[i+1]["time"]
            entry_price = float(df.iloc[i+1]["ask_o"] if side == "long" else df.iloc[i+1]["bid_o"])
            # We exit when signal flips or when TP/SL would be hit if your plugin provides them.
            # For simplicity (and to keep compatibility) we exit when signal flips back to 0 or opposite.
            for j in range(i+1, len(df)-1):
                s2 = int(sig.iloc[j])
                if (side == "long" and s2 <= 0) or (side == "short" and s2 >= 0):
                    exit_time = df.iloc[j+1]["time"]
                    exit_price = float(df.iloc[j+1]["bid_o"] if side == "long" else df.iloc[j+1]["ask_o"])
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                    })
                    pos = 0
                    break
            else:
                # force close on the last available bar
                exit_time = df.iloc[-1]["time"]
                exit_price = float(df.iloc[-1]["bid_c"] if side == "long" else df.iloc[-1]["ask_c"])
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                })
        pos = s

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {"symbol": symbol, "trades": 0, "win_rate": 0.0}
        return trades_df, pd.Series([start_equity]), summary

    # ISO timestamps for Excel friendliness
    trades_df["entry_time"] = trades_df["entry_time"].map(_iso)
    trades_df["exit_time"]  = trades_df["exit_time"].map(_iso)

    # equity + pnl
    eq_curve = _compute_equity(trades_df, start_equity, symbol)

    # summary
    pnl = trades_df["pnl"].values
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    wr = 100.0 * wins / max(1, wins + losses)
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": int(len(trades_df)),
        "win_rate": float(np.round(wr, 2)),
    }
    return trades_df, eq_curve, summary

def main():
    ap = argparse.ArgumentParser(description="Backtest and export CSVs.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--lookback", type=int, required=True)
    ap.add_argument("--trades", required=True, help="Path to write trades CSV")
    ap.add_argument("--equity", required=True, help="Path to write equity CSV")
    ap.add_argument("--start_equity", type=float, default=10000.0)
    ap.add_argument("--plugin_module", default="user_strategy.mean_reversion_mt5")
    ap.add_argument("--plugin_function", default="generate_signals")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.trades), exist_ok=True)
    os.makedirs(os.path.dirname(args.equity), exist_ok=True)

    trades_df, eq_curve, summary = backtest(
        args.symbol,
        args.timeframe,
        args.lookback,
        args.plugin_module,
        args.plugin_function,
        args.start_equity,
    )

    print(json.dumps(summary, indent=2))

    # save CSVs
    trades_df.to_csv(args.trades, index=False)
    # equity CSV with two cols: index (sequence) and equity value
    eq_df = pd.DataFrame({"step": list(eq_curve.index), "equity": eq_curve.values})
    eq_df.to_csv(args.equity, index=False)

    # nice console preview (first/last few rows)
    with pd.option_context("display.width", 140, "display.max_columns", 12):
        print("\n=== Trades head ===")
        print(trades_df.head(12))
        print("\n=== Trades tail ===")
        print(trades_df.tail(12))

if __name__ == "__main__":
    main()
