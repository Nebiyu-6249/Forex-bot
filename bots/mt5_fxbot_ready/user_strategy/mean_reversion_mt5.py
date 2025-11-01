# filename: user_strategy/mean_reversion_mt5.py
"""
Bollinger re-entry mean-reversion signal plugin (win-rate prioritized).
Returns a Series aligned to df.index: +1 (long), -1 (short), 0 (flat).
Execute on the next bar open. Size by ATR risk in the engine/backtest.
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

# =============================== TUNABLES =============================== #
# Win-rate > trade count: tightened vs earlier presets.
BB_N: int        = 18    # Bollinger period
BB_K: float      = 1.2 # Std-dev; higher => deeper extremes (fewer but cleaner)
ATR_N: int       = 14
ADX_N: int       = 14
ADX_MAX: float   = 29 # trend must be weak (lower is stricter)
SLOPE_MAX: float = 6.5e-05   # |slope(SMA200)| per bar (flatter is better)

# Only trade when ATR percentile is not too high (avoid top 50% most volatile bars)
ATR_PCTL_MAX: float = 0.95
# Require room to revert: distance to mid in ATRs
DIST_MIN_ATR: float = 0.15
DIST_MAX_ATR: float = 1.40

# Re-entry quality margins in ATRs
OUTSIDE_ATR: float = 0.086 # previous bar outside by >= this * ATR(prev)
INSIDE_ATR:  float = 0.04 # signal bar back inside by >= this * ATR(curr)

# Cooldown to avoid signal clusters
COOLDOWN: int = 0
# ====================================================================== #

# --------------------------- indicator helpers -------------------------- #
def _sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()

def _boll_mid_sd(x: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    mid = _sma(x, n)
    sd  = x.rolling(n, min_periods=n).std(ddof=0)
    return mid, sd

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up   = h.diff()
    down = -l.diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0),  up,   0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr  = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean()  / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def _slope(series: pd.Series, n: int = 20) -> pd.Series:
    if n <= 1:
        return series.diff()
    x = np.arange(n)
    x_mean = x.mean()
    denom = float(np.sum((x - x_mean)**2))
    def _calc(a: np.ndarray) -> float:
        y = a; y_mean = y.mean()
        num = np.sum((x - x_mean) * (y - y_mean))
        return 0.0 if denom == 0 else num / denom
    return series.rolling(n, min_periods=n).apply(lambda arr: _calc(np.asarray(arr)), raw=True)

def _rolling_percentile_of_last(x: pd.Series, window: int = 200) -> pd.Series:
    """Percentile rank of the last value inside each rolling window."""
    arr = x.to_numpy()
    out = np.full(len(arr), np.nan)
    if len(arr) < window:
        return pd.Series(out, index=x.index)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1 : i + 1]
        out[i] = np.sum(w <= w[-1]) / float(window)
    return pd.Series(out, index=x.index)

# --------------------------- column auto-pick ---------------------------- #
def _pick(df: pd.DataFrame, names: List[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n].astype(float)
    raise KeyError(f"Expected one of {names}, but got: {list(df.columns)}")

# ------------------------------- main API -------------------------------- #
def generate_signals(df: pd.DataFrame, timeframe: str | None = None) -> pd.Series:
    """
    Emit +1/-1/0 on the last CLOSED bar using Bollinger re-entry.
    Host executes at next bar open.
    """
    if df is None or df.empty:
        return pd.Series(dtype=int)

    c = _pick(df, ["mid_c", "close", "Close", "bid_c", "ask_c"])
    h = _pick(df, ["mid_h", "high", "High", "bid_h", "ask_h"])
    l = _pick(df, ["mid_l", "low",  "Low",  "bid_l", "ask_l"])

    mid, sd = _boll_mid_sd(c, BB_N)
    up = mid + BB_K * sd
    lo = mid - BB_K * sd
    atr = _atr(h, l, c, ATR_N)
    adx = _adx(h, l, c, ADX_N)
    slope200 = _slope(_sma(c, 200), 20)

    prank = _rolling_percentile_of_last(atr, 200).fillna(0.5)
    low_vol_ok = prank <= ATR_PCTL_MAX
    range_ok = (adx < ADX_MAX) & (slope200.abs() < SLOPE_MAX)

    # Re-entry quality: prior close beyond band by OUTSIDE_ATR*ATR(prev), back inside by INSIDE_ATR*ATR(curr)
    outside_long = (c.shift(1) <= lo.shift(1) - OUTSIDE_ATR * atr.shift(1))
    inside_long  = (c >= lo + INSIDE_ATR * atr)
    re_long = outside_long & inside_long

    outside_short = (c.shift(1) >= up.shift(1) + OUTSIDE_ATR * atr.shift(1))
    inside_short  = (c <= up - INSIDE_ATR * atr)
    re_short = outside_short & inside_short

    # Distance to mean (need room to revert)
    dist_long  = (mid - c)
    dist_short = (c - mid)
    dist_ok_long  = (dist_long  >= DIST_MIN_ATR * atr) & (dist_long  <= DIST_MAX_ATR * atr)
    dist_ok_short = (dist_short >= DIST_MIN_ATR * atr) & (dist_short <= DIST_MAX_ATR * atr)

    long_raw  = re_long  & dist_ok_long  & range_ok & low_vol_ok
    short_raw = re_short & dist_ok_short & range_ok & low_vol_ok
    raw_sig = np.where(long_raw, 1, np.where(short_raw, -1, 0)).astype(int)

    # Cooldown
    out = np.zeros_like(raw_sig)
    last = -10_000
    for i, v in enumerate(raw_sig):
        if v != 0 and (i - last) <= COOLDOWN:
            out[i] = 0
        else:
            out[i] = v
            if v != 0:
                last = i

    return pd.Series(out, index=df.index, dtype=int)
