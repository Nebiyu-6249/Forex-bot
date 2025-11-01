# filename: scripts/export_trades_with_equity.py
# -----------------------------------------------------------
# Proper equity simulation with risk-based sizing (+ optional scale-out).
# - Uses your existing backtest to get the trade list (unchanged entries/exits)
# - Sizes from risk.risk_per_trade (or fixed_lots) with broker-like bounds
# - If exits.scale_out_enable: true, books partial at +1R (mid-band proxy),
#   remainder at the actual exit, and flags 'scaled_out'
# - Exports side-by-side "no-scale" vs "with-scale" equity/lot paths
# -----------------------------------------------------------
import os
import sys
import math
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))  # .../scripts
sys.path.append(os.path.dirname(ROOT))             # project root

# We only read the trade list; backtest logic (exits) stays unchanged.
from mt5fx.backtest import run_backtest  # returns {"summary": ..., "trades": df}  # noqa: E402


# ---------- config helpers ----------

def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_iso(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


def _as_float(val, default: float) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _lot_bounds_from_cfg(risk_cfg: Dict[str, Any]) -> Tuple[float, float, float, bool]:
    """
    Parse min/max/step + rounding policy from risk config. Supports 'auto'.
    Defaults are broker-agnostic: vmin=0.01, step=0.01, vmax=100.0
    """
    vmin = risk_cfg.get("min_lot", "auto")
    vmax = risk_cfg.get("max_lot", "auto")
    step = risk_cfg.get("lot_step", "auto")  # optional
    round_to_step = bool(risk_cfg.get("round_to_step", True))

    vmin = 0.01 if (isinstance(vmin, str) and vmin.lower() == "auto") else _as_float(vmin, 0.01)
    vmax = 100.0 if (isinstance(vmax, str) and vmax.lower() == "auto") else _as_float(vmax, 100.0)
    step = 0.01 if (isinstance(step, str) and step.lower() == "auto") else _as_float(step, 0.01)

    if step <= 0:
        step = 0.01
    if vmin <= 0:
        vmin = 0.01
    if vmax < vmin:
        vmax = vmin
    return vmin, vmax, step, round_to_step


# ---------- market model helpers (simple & robust for majors) ----------

def _pip_meta(symbol: str) -> Tuple[float, float]:
    """
    Return (pip_size, pip_value_usd_per_1lot) for the symbol.
    Assumptions:
      - Most USD-quoted majors (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCHF, USDCAD): pip size 0.0001, $10 per 1.0 lot.
      - JPY crosses (e.g., USDJPY): pip size 0.01, $9-10 per 1.0 lot (we use $10 to be conservative).
    """
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01, 10.0
    return 0.0001, 10.0


def _round_to_step(lots: float, vmin: float, vmax: float, step: float, round_to_step: bool) -> float:
    lots = max(vmin, min(vmax, lots))
    if round_to_step:
        lots = math.floor(lots / step + 1e-12) * step
        prec = 0 if step >= 1 else max(0, int(round(-math.log10(step))))
        lots = round(lots, prec)
    return lots


def _risk_sized_lots(entry: float, sl: float, equity: float, risk_pct: float,
                     symbol: str, risk_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Compute lots to risk 'risk_pct * equity' if SL is hit.
    Returns (lots, risk_usd_at_sl, pips_to_sl).
    """
    pip_sz, pip_val = _pip_meta(symbol)
    dist = abs(entry - sl)
    if not np.isfinite(dist) or dist <= 0:
        return 0.0, 0.0, 0.0

    pips_to_sl = dist / pip_sz
    loss_per_lot = pips_to_sl * pip_val  # USD loss per 1.0 lot at SL

    risk_amt = max(0.0, float(equity) * float(risk_pct))
    if loss_per_lot <= 0 or risk_amt <= 0:
        return 0.0, 0.0, pips_to_sl

    raw_lots = risk_amt / loss_per_lot
    vmin, vmax, step, round_to_step = _lot_bounds_from_cfg(risk_cfg)
    lots = _round_to_step(raw_lots, vmin, vmax, step, round_to_step)

    risk_usd = lots * loss_per_lot
    return lots, risk_usd, pips_to_sl


def _pnl_usd(entry: float, exit_px: float, side: str, lots: float, symbol: str) -> float:
    pip_sz, pip_val = _pip_meta(symbol)
    pips = (exit_px - entry) / pip_sz if side == "long" else (entry - exit_px) / pip_sz
    return lots * pip_val * pips


def _scaleout_book(entry: float, exit_px: float, sl: float, side: str,
                   lots: float, frac: float, symbol: str) -> Tuple[float, bool, float]:
    """
    Book P&L with scale-out at +1R (mid-band proxy). Remainder exits at actual exit price.
    Returns: (pnl_with_scale, scaled_hit, rr_final)
    """
    if lots <= 0:
        return 0.0, False, 0.0

    R = abs(entry - sl)
    if not np.isfinite(R) or R <= 0:
        # No sensible R, fall back to no-scale P&L
        pnl = _pnl_usd(entry, exit_px, side, lots, symbol)
        return pnl, False, 0.0

    # 1R level from entry (proxy for "touch mid-band")
    mid = entry + R if side == "long" else entry - R

    # Did we plausibly reach +1R by the time we exited? (We only have exit_px here.)
    scaled_hit = (exit_px >= mid) if side == "long" else (exit_px <= mid)

    # Realized R multiple (for diagnostics)
    rr_final = ((exit_px - entry) / R) if side == "long" else ((entry - exit_px) / R)

    if not scaled_hit:
        return _pnl_usd(entry, exit_px, side, lots, symbol), False, rr_final

    pnl_part1 = _pnl_usd(entry, mid, side, lots * frac, symbol)             # close 'frac' at +1R
    pnl_part2 = _pnl_usd(entry, exit_px, side, lots * (1.0 - frac), symbol) # remainder at exit
    return pnl_part1 + pnl_part2, True, rr_final


# ---------- main equity pass (portfolio-specific sizing) ----------

def enrich_with_equity_and_scaleout(trades: pd.DataFrame, cfg: dict, start_equity: float) -> pd.DataFrame:
    """
    For each trade:
      - Size lots from the *current* equity of each portfolio (no-scale vs with-scale)
      - Compute pnl_usd_noscale / pnl_usd_withscale
      - Accumulate equity_ns / equity (with-scale)
      - Log whether scale-out was hit (scaled_out) and final R multiple (rr_final)
    """
    if trades is None or trades.empty:
        return trades

    g = cfg["general"];   symbol = g["symbol"]
    ex = cfg.get("exits", {}); risk_cfg = cfg.get("risk", {})

    scale_on   = bool(ex.get("scale_out_enable", False))
    scale_frac = float(ex.get("scale_out_fraction", 0.5))
    risk_pct   = float(risk_cfg.get("risk_per_trade", 0.0))
    fixed_lots = float(risk_cfg.get("fixed_lots", 0.0))

    df = trades.copy().reset_index(drop=True)
    # Spreadsheet-friendly times
    if "entry_time" in df.columns: df["entry_time"] = _to_iso(df["entry_time"])
    if "exit_time"  in df.columns: df["exit_time"]  = _to_iso(df["exit_time"])

    # Working arrays
    lots_ns, risk_usd_ns, pnl_ns_arr, eq_ns_arr = [], [], [], []
    lots_ws, risk_usd_ws, pnl_ws_arr, eq_ws_arr = [], [], [], []
    rr_final_arr, scaled_flag_arr = [], []

    eq_ns = float(start_equity)
    eq_ws = float(start_equity)

    for _, row in df.iterrows():
        side   = str(row["side"]).lower()
        entry  = float(row["entry_price"])
        exit_px = float(row["exit_price"])
        sl     = row.get("sl", np.nan)
        sl     = float(sl) if np.isfinite(sl) else (entry - 0.001 if side == "long" else entry + 0.001)

        # --- sizing for the "no-scale" portfolio ---
        if fixed_lots > 0:
            l_ns, r_ns, _ = fixed_lots, np.nan, np.nan
        else:
            l_ns, r_ns, _ = _risk_sized_lots(entry, sl, eq_ns, risk_pct, symbol, risk_cfg)

        pnl_ns = _pnl_usd(entry, exit_px, side, l_ns, symbol)
        eq_ns += pnl_ns

        lots_ns.append(l_ns); risk_usd_ns.append(r_ns); pnl_ns_arr.append(pnl_ns); eq_ns_arr.append(eq_ns)

        # --- sizing for the "with-scale" portfolio (may differ) ---
        if fixed_lots > 0:
            l_ws, r_ws, _ = fixed_lots, np.nan, np.nan
        else:
            l_ws, r_ws, _ = _risk_sized_lots(entry, sl, eq_ws, risk_pct, symbol, risk_cfg)

        if scale_on:
            pnl_ws, scaled_hit, rr_final = _scaleout_book(entry, exit_px, sl, side, l_ws, scale_frac, symbol)
        else:
            pnl_ws = _pnl_usd(entry, exit_px, side, l_ws, symbol)
            scaled_hit = False
            R = abs(entry - sl); rr_final = ((exit_px - entry) / R) if side == "long" else ((entry - exit_px) / R) if R > 0 else 0.0

        eq_ws += pnl_ws

        lots_ws.append(l_ws); risk_usd_ws.append(r_ws); pnl_ws_arr.append(pnl_ws); eq_ws_arr.append(eq_ws)
        rr_final_arr.append(rr_final); scaled_flag_arr.append(bool(scaled_hit))

    # Emit columns
    df["lots_ns"]        = lots_ns
    df["risk_usd_ns"]    = risk_usd_ns
    df["pnl_usd_noscale"]= pnl_ns_arr
    df["equity_ns"]      = eq_ns_arr

    df["lots_ws"]        = lots_ws
    df["risk_usd_ws"]    = risk_usd_ws
    df["pnl_usd_withscale"] = pnl_ws_arr
    df["equity"]         = eq_ws_arr            # main equity (respects exits.scale_out_enable)

    df["scaled_out"]     = scaled_flag_arr
    df["rr_final"]       = rr_final_arr

    return df


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Export trades CSV with risk-based equity (+optional scale-out).")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--lookback", type=int)
    ap.add_argument("--equity", type=float, default=10_000.0, help="Starting equity in USD")
    ap.add_argument("--out", default="out/trades.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = _load_cfg(args.config)
    # CLI overrides
    if args.symbol:    cfg["general"]["symbol"] = args.symbol
    if args.timeframe: cfg["general"]["timeframe"] = args.timeframe
    if args.lookback:  cfg.setdefault("backtest", {}); cfg["backtest"]["lookback"] = args.lookback

    # Use YOUR backtest for the trade list (unchanged logic/exits).
    res = run_backtest(cfg)  # returns {'summary': ..., 'trades': df}
    trades_df = res.get("trades", pd.DataFrame())

    enriched = enrich_with_equity_and_scaleout(trades_df, cfg, start_equity=args.equity)
    enriched.to_csv(args.out, index=False)

    # console summary
    wins = float((enriched["pnl_usd_withscale"] > 0).mean() * 100.0) if not enriched.empty else 0.0
    print(f"Saved {len(enriched)} trades to {args.out}")
    print(f"Win rate: {wins:.2f}%")
    if not enriched.empty:
        print(f"Final equity: {enriched['equity'].iloc[-1]:.2f}")
        print(f"Scale-outs hit: {int(enriched['scaled_out'].sum())} of {len(enriched)}")


if __name__ == "__main__":
    main()
