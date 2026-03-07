# mt5fx/multi_tf.py
"""
Multi-timeframe confirmation module.

Fetches higher-timeframe data to confirm or block signals from the primary timeframe.
Key principle: never take a mean-reversion long on M15 if H4 is in a strong downtrend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import MetaTrader5 as mt5


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _slope(series: pd.Series, n: int = 10) -> pd.Series:
    return series.diff(n)


def get_htf_trend(
    symbol: str,
    tf_enum: int,
    ema_fast: int = 50,
    ema_slow: int = 200,
    lookback: int = 300,
) -> dict:
    """
    Fetch higher-timeframe data and compute trend direction.

    Returns dict with:
      - direction: +1 (bullish), -1 (bearish), 0 (neutral/flat)
      - strength: 0.0 to 1.0 (how strong the trend is)
      - ema_fast: current fast EMA value
      - ema_slow: current slow EMA value
    """
    rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, lookback)
    if rates is None or len(rates) < ema_slow + 10:
        return {"direction": 0, "strength": 0.0, "ema_fast": 0.0, "ema_slow": 0.0}

    df = pd.DataFrame(rates)
    c = df["close"].astype(float)

    fast = _ema(c, ema_fast)
    slow = _ema(c, ema_slow)

    fast_val = float(fast.iloc[-1])
    slow_val = float(slow.iloc[-1])

    # Direction based on EMA relationship
    if fast_val > slow_val:
        direction = 1
    elif fast_val < slow_val:
        direction = -1
    else:
        direction = 0

    # Strength: normalized distance between fast and slow EMA
    if slow_val > 0:
        strength = min(1.0, abs(fast_val - slow_val) / slow_val * 100)
    else:
        strength = 0.0

    # Additional: is the trend accelerating or decelerating?
    fast_slope = _slope(fast, 5)
    if len(fast_slope) > 0 and not np.isnan(fast_slope.iloc[-1]):
        slope_sign = 1 if fast_slope.iloc[-1] > 0 else -1
        # If slope disagrees with direction, reduce strength
        if slope_sign != direction:
            strength *= 0.5

    return {
        "direction": direction,
        "strength": strength,
        "ema_fast": fast_val,
        "ema_slow": slow_val,
    }


def should_block_signal(
    signal: int,
    symbol: str,
    confirmation_tf: int,
    trend_tf: int,
    ema_fast: int = 50,
    ema_slow: int = 200,
    block_counter_trend: bool = True,
) -> tuple[bool, str]:
    """
    Check if a signal should be blocked based on higher-timeframe context.

    Args:
        signal: +1 (long) or -1 (short) from primary strategy
        symbol: trading symbol
        confirmation_tf: MT5 timeframe enum for confirmation (e.g., H1)
        trend_tf: MT5 timeframe enum for trend (e.g., H4)
        ema_fast/slow: EMA periods for trend detection
        block_counter_trend: if True, block signals against the HTF trend

    Returns:
        (should_block, reason)
    """
    if signal == 0 or not block_counter_trend:
        return False, ""

    # Check trend on the higher timeframe
    htf_trend = get_htf_trend(symbol, trend_tf, ema_fast, ema_slow)

    # Strong counter-trend signal: block it
    if htf_trend["strength"] > 0.3:  # meaningful trend
        if signal > 0 and htf_trend["direction"] < 0:
            return True, f"blocked long: H4 bearish (strength={htf_trend['strength']:.2f})"
        if signal < 0 and htf_trend["direction"] > 0:
            return True, f"blocked short: H4 bullish (strength={htf_trend['strength']:.2f})"

    # Also check confirmation timeframe for alignment
    conf_trend = get_htf_trend(symbol, confirmation_tf, ema_fast, ema_slow)
    if conf_trend["strength"] > 0.2:
        if signal > 0 and conf_trend["direction"] < 0:
            return True, f"blocked long: H1 bearish (strength={conf_trend['strength']:.2f})"
        if signal < 0 and conf_trend["direction"] > 0:
            return True, f"blocked short: H1 bullish (strength={conf_trend['strength']:.2f})"

    return False, ""


def get_htf_trend_from_df(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> int:
    """
    Compute trend direction from a DataFrame (for backtesting without MT5).
    Returns +1 (bullish), -1 (bearish), 0 (neutral).
    """
    c = df["mid_c"] if "mid_c" in df.columns else df.get("bid_c", df.get("close", pd.Series()))
    if len(c) < ema_slow + 10:
        return 0

    fast = _ema(c, ema_fast)
    slow = _ema(c, ema_slow)

    fast_val = float(fast.iloc[-1])
    slow_val = float(slow.iloc[-1])

    if fast_val > slow_val:
        return 1
    elif fast_val < slow_val:
        return -1
    return 0
