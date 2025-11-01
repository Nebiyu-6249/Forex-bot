# user_strategy/my_alpha.py
import numpy as np
import pandas as pd

# ---- Tunables (edit these to tune behaviour) ----
FAST = 20
SLOW = 50
TREND = 200

ADX_PERIOD = 14
ADX_MIN = 22.0          # ← start at 22; try 20–25 range

ATR_PERIOD = 14
ATR_LOOKBACK = 200
ATR_QUANTILE = 0.25     # ← volume filter; try 0.20–0.35

# Session hours you want to trade, expressed in the **target timezone**
SESSION_HOURS = set(range(7, 20))  # 07:00–19:59
# Offset to convert broker-server timestamps to your target timezone.
# Example: broker = UTC+2 → set SESSION_OFFSET_HOURS = +2 to convert to UTC.
SESSION_OFFSET_HOURS = +2          # ← adjust to your broker; try +2 or +3

COOLDOWN_BARS = 8                  # reduce clustering without starving entries
TREND_SLOPE_BARS = 8               # slope lookback for 200-EMA confirmation
DIST_K = 0.20                      # min distance from 200-EMA in ATRs, e.g., 0.2*ATR

def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff(); down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, n)
    plus_di  = 100.0 * rma(pd.Series(plus_dm, index=high.index), n) / atr
    minus_di = 100.0 * rma(pd.Series(minus_dm, index=high.index), n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx, n)

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]; high = df["High"]; low = df["Low"]
    # Baseline cross + trend
    ema_fast  = close.ewm(span=FAST, adjust=False).mean()
    ema_slow  = close.ewm(span=SLOW, adjust=False).mean()
    ema_trend = close.ewm(span=TREND, adjust=False).mean()
    cross_up   = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    cross_down = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    # Trend strength + volatility gate
    adx_val = adx(high, low, close, ADX_PERIOD)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_PERIOD)
    atr_thresh = atr.rolling(ATR_LOOKBACK, min_periods=ATR_LOOKBACK // 2).quantile(ATR_QUANTILE)
    vol_ok = atr > atr_thresh

    # Session filter (convert broker server time → target clock)
    if "time" in df.columns:
        t = df["time"]
        if SESSION_OFFSET_HOURS:
            t = t - pd.to_timedelta(SESSION_OFFSET_HOURS, unit="h")
        session_ok = t.dt.hour.isin(SESSION_HOURS)
    else:
        session_ok = pd.Series(True, index=df.index)

    # Extra quality gates: slope of 200-EMA + distance from trend
    slope = ema_trend - ema_trend.shift(TREND_SLOPE_BARS)
    long_ok  = cross_up   & (close > ema_trend) & (slope > 0) & (adx_val >= ADX_MIN) & vol_ok & session_ok & ((close - ema_trend) > DIST_K*atr)
    short_ok = cross_down & (close < ema_trend) & (slope < 0) & (adx_val >= ADX_MIN) & vol_ok & session_ok & ((ema_trend - close) > DIST_K*atr)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_ok]  = 1
    sig[short_ok] = -1

    # Cooldown: suppress new signals for N bars after any nonzero
    if COOLDOWN_BARS > 0:
        recent = sig.ne(0).rolling(COOLDOWN_BARS).max().shift(1).fillna(0).astype(bool)
        sig = sig.where(~recent, 0)

    return sig
