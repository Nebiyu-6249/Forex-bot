# mt5fx/walk_forward.py
"""
Walk-forward validation for the backtester.

Splits historical data into rolling train/test windows to detect overfitting.
Reports out-of-sample performance separately from in-sample.
"""

from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .backtest import backtest_once


def walk_forward_validation(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    train_bars: int = 4000,
    test_bars: int = 1000,
    step_bars: int = 1000,
) -> Dict[str, Any]:
    """
    Run walk-forward validation.

    Args:
        cfg: full config dict
        df: full DataFrame of price data
        train_bars: number of bars for training window
        test_bars: number of bars for testing window
        step_bars: how many bars to step forward each iteration

    Returns:
        Dict with:
          - windows: list of per-window results
          - oos_summary: aggregated out-of-sample statistics
    """
    total_bars = len(df)
    if total_bars < train_bars + test_bars:
        return {
            "error": f"Not enough data: {total_bars} bars, need {train_bars + test_bars}",
            "windows": [],
            "oos_summary": {},
        }

    windows: List[Dict[str, Any]] = []
    all_oos_trades = []

    start = 0
    while start + train_bars + test_bars <= total_bars:
        train_end = start + train_bars
        test_end = train_end + test_bars

        train_df = df.iloc[start:train_end].reset_index(drop=True)
        test_df = df.iloc[train_end:test_end].reset_index(drop=True)

        # Run backtest on training set (in-sample)
        is_result = backtest_once(cfg, train_df)
        # Run backtest on test set (out-of-sample)
        oos_result = backtest_once(cfg, test_df)

        window = {
            "window": len(windows) + 1,
            "train_start": str(train_df.iloc[0]["time"]) if "time" in train_df.columns else start,
            "train_end": str(train_df.iloc[-1]["time"]) if "time" in train_df.columns else train_end,
            "test_start": str(test_df.iloc[0]["time"]) if "time" in test_df.columns else train_end,
            "test_end": str(test_df.iloc[-1]["time"]) if "time" in test_df.columns else test_end,
            "is_win_rate": round(is_result.get("win_rate", 0) * 100, 2),
            "is_trades": len(is_result.get("trades", [])),
            "is_pf": round(is_result.get("profit_factor", 0), 2),
            "oos_win_rate": round(oos_result.get("win_rate", 0) * 100, 2),
            "oos_trades": len(oos_result.get("trades", [])),
            "oos_pf": round(oos_result.get("profit_factor", 0), 2),
            "oos_total_pnl": round(oos_result.get("total_pnl", 0), 2),
        }
        windows.append(window)

        oos_trades_df = oos_result.get("trades", pd.DataFrame())
        if not oos_trades_df.empty:
            all_oos_trades.append(oos_trades_df)

        start += step_bars

    # Aggregate out-of-sample stats
    oos_summary = {}
    if all_oos_trades:
        combined = pd.concat(all_oos_trades, ignore_index=True)
        if not combined.empty and "pnl_usd" in combined.columns:
            winners = combined[combined["pnl_usd"] > 0]
            losers = combined[combined["pnl_usd"] <= 0]
            gross_profit = float(winners["pnl_usd"].sum()) if not winners.empty else 0.0
            gross_loss = abs(float(losers["pnl_usd"].sum())) if not losers.empty else 0.0

            oos_summary = {
                "total_trades": len(combined),
                "win_rate": round(float((combined["pnl_usd"] > 0).mean()) * 100, 2),
                "total_pnl": round(float(combined["pnl_usd"].sum()), 2),
                "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
                "expectancy": round(float(combined["pnl_usd"].mean()), 2),
                "windows_profitable": sum(1 for w in windows if w["oos_total_pnl"] > 0),
                "windows_total": len(windows),
                "consistency": round(sum(1 for w in windows if w["oos_total_pnl"] > 0) / len(windows) * 100, 1) if windows else 0,
            }

    return {
        "windows": windows,
        "oos_summary": oos_summary,
    }
