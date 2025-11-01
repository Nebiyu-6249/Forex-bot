# mt5fx/backtest.py
# -----------------------------------------------------------------------------
# Backtester with risk-based sizing + mid-band scale-out (parity with LIVE)
# - ATR-based SL/TP (as before)
# - Breakeven stop after +R move (breakeven_at_R)
# - ATR trailing (trail_atr_mult)
# - Optional time-stop (max_hold_bars)
# - "Pessimistic" intrabar resolution when both SL & TP touch same bar
# - NEW: risk.risk_per_trade sizing (compounding) OR risk.fixed_lots
# - NEW: exits.scale_out_* logic realized in PnL and flagged per trade
# - Outputs pnl_usd, size_lots, scaled_out (so CSV shows the effect directly)
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import os

import numpy as np
import pandas as pd

from .data import load_rates
from .strategy import ema_cross_with_trend, attach_atr
from .strategy_loader import compute_plugin_signal


# -----------------------------------------------------------------------------


def _bar_hit(side: str, row: pd.Series, tp: float, sl: float, intrabar_mode: str = "pessimistic"):
    """
    Decide which exit was hit within 'row' bar (uses bar OHLC and bid/ask logic).
    'pessimistic' -> if both TP & SL are touched on same bar, count it as SL.
    """
    if side == "long":
        tp_touched = row["bid_h"] >= tp
        sl_touched = row["bid_l"] <= sl
    else:
        tp_touched = row["ask_l"] <= tp
        sl_touched = row["ask_h"] >= sl

    if tp_touched and sl_touched:
        return "sl" if intrabar_mode == "pessimistic" else "tp"
    if tp_touched:
        return "tp"
    if sl_touched:
        return "sl"
    return None


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


# -----------------------------------------------------------------------------


@dataclass
class TradeRow:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    result: str           # 'tp' | 'sl' | 'time' | 'be_trail'
    size_lots: float
    pnl_usd: float
    scaled_out: bool


@dataclass
class Pos:
    side: str
    entry: float
    sl: float
    tp: float
    time: pd.Timestamp
    lots: float
    r_price: float              # R in price units (abs(entry - sl))
    be_applied: bool = False
    bars_open: int = 0
    scaled_out: bool = False
    partial_lots: float = 0.0   # lots closed at scale-out
    partial_usd: float = 0.0    # realized cash at scale-out


# -----------------------------------------------------------------------------


def backtest_once(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    gcfg = cfg.get("general", {})
    s = cfg["strategy"]
    xcfg = cfg.get("exits", {})
    rcfg = cfg.get("risk", {})
    bcfg = cfg.get("backtest", {})

    # --- EXITS CONFIG (mirrors engine.py) ---
    be_on = bool(xcfg.get("breakeven_enable", False))
    be_R = float(xcfg.get("breakeven_at_R", 0.6))
    trail_on = bool(xcfg.get("trail_enable", False))
    trail_k = float(xcfg.get("trail_atr_mult", 1.0))
    max_hold = int(xcfg.get("max_hold_bars", 0))
    intrabar_mode = xcfg.get("intrabar_mode", "pessimistic")

    # Scale-out
    scale_on = bool(xcfg.get("scale_out_enable", False))
    scale_frac = float(xcfg.get("scale_out_fraction", 0.50))
    rr_after = float(xcfg.get("rr_after_scale", 3.0))
    bb_n = int(xcfg.get("bb_period_for_mean", 20))

    # RISK / SIZING
    risk_pct = float(rcfg.get("risk_per_trade", 0.0))
    fixed_lots = float(rcfg.get("fixed_lots", 0.0))
    lot_step = float(rcfg.get("lot_step", 0.01)) if "lot_step" in rcfg else 0.01
    min_lot = rcfg.get("min_lot", "auto")
    min_lot = 0.01 if (min_lot == "auto") else float(min_lot)
    max_lot = rcfg.get("max_lot", "auto")
    max_lot = 100.0 if (max_lot == "auto") else float(max_lot)
    # Assume $10 per pip per lot for USD-quoted majors, overrideable via config
    pip_value_per_lot = float(rcfg.get("pip_value_per_lot_usd", 10.0))

    # STARTING EQUITY for compounding (provided by exporter)
    equity = float(bcfg.get("start_equity", 10_000.0))

    # --- set env for profile-aware plugins ---
    os.environ["BOT_SYMBOL"] = str(gcfg.get("symbol", ""))
    os.environ["BOT_TIMEFRAME"] = str(gcfg.get("timeframe", ""))

    # --- signals ---
    if s.get("use_plugin", False):
        sig = compute_plugin_signal(
            df,
            s["plugin_module"],
            s.get("plugin_function", "generate_signals"),
            s.get("plugin_shift_next_bar", True),
        )
    else:
        sig = ema_cross_with_trend(df, fast=s["fast_ema"], slow=s["slow_ema"], trend=s["trend_ema"])

    # --- attach ATR and compute mid-band for scale-out ---
    df = attach_atr(df, n=s["atr_period"])
    mid_c = (_sma((df["bid_c"] + df["ask_c"]) / 2.0, bb_n)).rename("mid")

    def _round_lots(x: float) -> float:
        x = max(min_lot, min(max_lot, (np.floor(x / lot_step) * lot_step)))
        prec = 0 if lot_step >= 1 else max(0, int(round(-np.log10(lot_step))))
        return round(x, prec)

    trades: List[TradeRow] = []
    pos: Pos | None = None

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        signal = int(sig.iloc[i])
        atr = float(row["ATR"])

        # -------- manage open position --------
        if pos is not None:
            pos.bars_open += 1

            # 0) scale-out check (once)
            if scale_on and (not pos.scaled_out) and not np.isnan(mid_c.iloc[i]):
                mid_val = float(mid_c.iloc[i])
                if pos.side == "long" and row["bid_h"] >= mid_val:
                    # scale at mid
                    part = _round_lots(pos.lots * scale_frac)
                    if part > 0:
                        pnl_pips = (mid_val - pos.entry) * 10000.0
                        pos.partial_usd += part * pip_value_per_lot * pnl_pips
                        pos.partial_lots += part
                        pos.lots -= part
                        # move TP of remainder to rr_after * R
                        pos.tp = pos.entry + rr_after * pos.r_price
                        pos.scaled_out = True

                elif pos.side == "short" and row["ask_l"] <= mid_val:
                    part = _round_lots(pos.lots * scale_frac)
                    if part > 0:
                        pnl_pips = (pos.entry - mid_val) * 10000.0
                        pos.partial_usd += part * pip_value_per_lot * pnl_pips
                        pos.partial_lots += part
                        pos.lots -= part
                        pos.tp = pos.entry - rr_after * pos.r_price
                        pos.scaled_out = True

            # current R based on original risk distance
            R = pos.r_price

            # 1) Breakeven
            if be_on and not pos.be_applied and R > 0:
                be_level = pos.entry + be_R * R if pos.side == "long" else pos.entry - be_R * R
                price_c = float(row["bid_c"] if pos.side == "long" else row["ask_c"])
                if (pos.side == "long" and price_c >= be_level) or (pos.side == "short" and price_c <= be_level):
                    if (pos.side == "long" and pos.entry > pos.sl) or (pos.side == "short" and pos.entry < pos.sl):
                        pos.sl = pos.entry
                        pos.be_applied = True

            # 2) ATR trailing (tightens only)
            if trail_on and not np.isnan(atr) and atr > 0:
                if pos.side == "long":
                    cand = float(row["bid_c"]) - trail_k * atr
                    if cand > pos.sl:
                        pos.sl = cand
                else:
                    cand = float(row["ask_c"]) + trail_k * atr
                    if cand < pos.sl:
                        pos.sl = cand

            # 3) Time-stop
            if max_hold and pos.bars_open >= max_hold:
                exit_price = float(row["bid_o"] if pos.side == "long" else row["ask_o"])
                # remainder PnL
                pnl_pips = ((exit_price - pos.entry) if pos.side == "long" else (pos.entry - exit_price)) * 10000.0
                remainder_usd = pos.lots * pip_value_per_lot * pnl_pips
                pnl_usd = pos.partial_usd + remainder_usd

                trades.append(
                    TradeRow(
                        entry_time=str(pos.time),
                        exit_time=str(row["time"]),
                        side=pos.side,
                        entry_price=pos.entry,
                        exit_price=exit_price,
                        sl=pos.sl,
                        tp=pos.tp,
                        result="time",
                        size_lots=pos.lots + pos.partial_lots,
                        pnl_usd=float(pnl_usd),
                        scaled_out=bool(pos.scaled_out),
                    )
                )
                equity += pnl_usd
                pos = None
            else:
                # normal bar hit evaluation
                hit = _bar_hit(pos.side, row, pos.tp, pos.sl, intrabar_mode=intrabar_mode)
                if hit:
                    exit_price = float(row["bid_o"] if pos.side == "long" else row["ask_o"])
                    tag = "be_trail" if (pos.be_applied or trail_on) and hit == "sl" else hit

                    pnl_pips = ((exit_price - pos.entry) if pos.side == "long" else (pos.entry - exit_price)) * 10000.0
                    remainder_usd = pos.lots * pip_value_per_lot * pnl_pips
                    pnl_usd = pos.partial_usd + remainder_usd

                    trades.append(
                        TradeRow(
                            entry_time=str(pos.time),
                            exit_time=str(row["time"]),
                            side=pos.side,
                            entry_price=pos.entry,
                            exit_price=exit_price,
                            sl=pos.sl,
                            tp=pos.tp,
                            result=tag,
                            size_lots=pos.lots + pos.partial_lots,
                            pnl_usd=float(pnl_usd),
                            scaled_out=bool(pos.scaled_out),
                        )
                    )
                    equity += pnl_usd
                    pos = None

        # -------- open new position --------
        if pos is None and signal != 0 and not np.isnan(atr) and atr > 0:
            side = "long" if signal > 0 else "short"
            entry = float(row["ask_o"] if side == "long" else row["bid_o"])

            use_custom = bool(s.get("use_custom_sl_tp", False))
            if use_custom and s.get("custom_exit_mode", "") == "prev_bar_extreme":
                prev = df.iloc[i - 1]
                if side == "long":
                    base_sl = float(prev["bid_l"])
                    risk = entry - base_sl
                else:
                    base_sl = float(prev["ask_h"])
                    risk = base_sl - entry
                min_buf = float(s.get("min_sl_buffer_atr", 0.0)) * float(atr)
                if risk <= max(min_buf, 1e-8):
                    if side == "long":
                        base_sl = entry - max(min_buf, 1e-6)
                        risk = entry - base_sl
                    else:
                        base_sl = entry + max(min_buf, 1e-6)
                        risk = base_sl - entry
                mult = float(s.get("tp_multiple", 2.0))
                sl = base_sl
                tp = entry + mult * risk if side == "long" else entry - mult * risk
            else:
                sl = entry - s["atr_sl_mult"] * atr if side == "long" else entry + s["atr_sl_mult"] * atr
                tp = entry + s["atr_tp_mult"] * atr if side == "long" else entry - s["atr_tp_mult"] * atr

            # ---- position sizing ----
            r_price = abs(entry - sl)
            r_pips = r_price * 10000.0
            lots = 0.0
            if fixed_lots > 0:
                lots = _round_lots(fixed_lots)
            else:
                if r_pips > 0 and risk_pct > 0:
                    risk_usd = equity * risk_pct
                    lots = _round_lots(risk_usd / (pip_value_per_lot * r_pips))
                else:
                    lots = _round_lots(min_lot)

            # if sizing is zero (e.g., too tight SL), skip
            if lots <= 0:
                continue

            pos = Pos(side=side, entry=entry, sl=sl, tp=tp, time=df.iloc[i]["time"], lots=lots, r_price=r_price)

    tr_df = pd.DataFrame([t.__dict__ for t in trades])
    win = float((tr_df["result"] == "tp").mean()) if not tr_df.empty else 0.0

    return {
        "trades": tr_df,          # includes pnl_usd, size_lots, scaled_out
        "win_rate": win,
    }


def run_backtest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    g = cfg["general"]
    b = cfg["backtest"]

    # ensure env available for profile-aware plugins
    os.environ["BOT_SYMBOL"] = g["symbol"]
    os.environ["BOT_TIMEFRAME"] = g["timeframe"]

    df = load_rates(g["symbol"], g["timeframe"], count=b["lookback"])
    res = backtest_once(cfg, df)

    summary = {
        "symbol": g["symbol"],
        "timeframe": g["timeframe"],
        "trades": int(len(res["trades"])),
        "win_rate": round(res["win_rate"] * 100, 2),
    }
    return {"summary": summary, "trades": res["trades"]}
