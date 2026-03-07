# mt5fx/regime_detector.py
"""
Market regime detection: classifies the current market as
TRENDING, RANGING, or CHOPPY based on ADX, Bollinger bandwidth, and ATR trend.

This allows automatic strategy selection:
  - RANGING  -> mean reversion strategies
  - TRENDING -> trend following strategies
  - CHOPPY   -> no trading (sit out)
"""

from __future__ import annotations
from enum import Enum

import numpy as np
import pandas as pd


class Regime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CHOPPY = "choppy"


def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_val
    dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _bb_width_percentile(c: pd.Series, bb_n: int = 20, lookback: int = 100) -> pd.Series:
    """Bollinger bandwidth percentile over lookback window."""
    mid = _sma(c, bb_n)
    sd = c.rolling(bb_n, min_periods=bb_n).std(ddof=0)
    width = (2 * sd / mid).replace(0, np.nan)

    arr = width.to_numpy()
    out = np.full(len(arr), np.nan)
    for i in range(lookback - 1, len(arr)):
        w = arr[i - lookback + 1: i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) > 0:
            out[i] = np.sum(valid <= valid[-1]) / float(len(valid))
    return pd.Series(out, index=c.index)


def detect_regime(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25,
    adx_choppy_threshold: float = 15,
    bb_width_ranging_pctl: float = 30,
    lookback_bars: int = 100,
) -> Regime:
    """
    Detect current market regime from the last bar of df.

    Logic:
    - ADX > trending_threshold AND rising -> TRENDING
    - ADX < choppy_threshold AND BB width very narrow -> CHOPPY (no direction, no range)
    - Otherwise (low ADX, moderate BB width) -> RANGING (good for mean reversion)
    """
    if df is None or len(df) < lookback_bars:
        return Regime.RANGING  # default to ranging if not enough data

    # Pick columns
    c = df["mid_c"] if "mid_c" in df.columns else df.get("bid_c", df.get("close", pd.Series()))
    h = df["mid_h"] if "mid_h" in df.columns else df.get("bid_h", df.get("high", pd.Series()))
    l = df["mid_l"] if "mid_l" in df.columns else df.get("bid_l", df.get("low", pd.Series()))

    adx_vals = _adx(h, l, c, 14)
    bb_pctl = _bb_width_percentile(c, 20, lookback_bars)

    current_adx = float(adx_vals.iloc[-1]) if not np.isnan(adx_vals.iloc[-1]) else 20.0
    current_bb_pctl = float(bb_pctl.iloc[-1]) if not np.isnan(bb_pctl.iloc[-1]) else 50.0

    # ADX rising = last 5 bars trend
    adx_rising = False
    if len(adx_vals) >= 5:
        recent = adx_vals.iloc[-5:]
        adx_rising = float(recent.iloc[-1]) > float(recent.iloc[0])

    # Classification
    if current_adx > adx_trending_threshold and adx_rising:
        return Regime.TRENDING
    elif current_adx < adx_choppy_threshold and current_bb_pctl * 100 < bb_width_ranging_pctl:
        return Regime.CHOPPY
    else:
        return Regime.RANGING


def detect_regime_series(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25,
    adx_choppy_threshold: float = 15,
    bb_width_ranging_pctl: float = 30,
    lookback_bars: int = 100,
) -> pd.Series:
    """
    Return a Series of regime labels for each bar (for backtesting).
    """
    c = df["mid_c"] if "mid_c" in df.columns else df.get("bid_c", df.get("close", pd.Series()))
    h = df["mid_h"] if "mid_h" in df.columns else df.get("bid_h", df.get("high", pd.Series()))
    l = df["mid_l"] if "mid_l" in df.columns else df.get("bid_l", df.get("low", pd.Series()))

    adx_vals = _adx(h, l, c, 14)
    bb_pctl = _bb_width_percentile(c, 20, lookback_bars)

    # ADX slope (5-bar)
    adx_rising = adx_vals.diff(5) > 0

    regimes = []
    for i in range(len(df)):
        adx_v = float(adx_vals.iloc[i]) if not np.isnan(adx_vals.iloc[i]) else 20.0
        bb_v = float(bb_pctl.iloc[i]) if not np.isnan(bb_pctl.iloc[i]) else 0.5
        rising = bool(adx_rising.iloc[i]) if not np.isnan(adx_rising.iloc[i]) else False

        if adx_v > adx_trending_threshold and rising:
            regimes.append(Regime.TRENDING.value)
        elif adx_v < adx_choppy_threshold and bb_v * 100 < bb_width_ranging_pctl:
            regimes.append(Regime.CHOPPY.value)
        else:
            regimes.append(Regime.RANGING.value)

    return pd.Series(regimes, index=df.index)
