# user_strategy/my_alpha_profiles.py
# -----------------------------------------------------------------------------
# FIX #3: A profile-aware version of "my_alpha".
# It reads BOT_SYMBOL/BOT_TIMEFRAME from environment (set in backtest/engine)
# and selects tuned filter constants per pair/timeframe.
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

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

# ---- Profiles (symbol, timeframe) -> constants
# NOTE: Session offset = +2 hours (your broker is GMT+2). Adjust if broker changes.
COMMON = dict(FAST=20, SLOW=50, TREND=200, ADX_PERIOD=14,
              SESSION_HOURS=set(range(7, 20)), SESSION_OFFSET_HOURS=+2,
              TREND_SLOPE_BARS=8)

PROFILES = {
    ("EURUSD", "M15"): dict(ADX_MIN=22, ATR_QUANTILE=0.3, COOLDOWN_BARS=8,  DIST_K=0.25, **COMMON),
    ("EURJPY", "M15"): dict(ADX_MIN=25, ATR_QUANTILE=0.30, COOLDOWN_BARS=8,  DIST_K=0.28, **COMMON),
    ("NZDUSD", "H1"):  dict(ADX_MIN=21, ATR_QUANTILE=0.22, COOLDOWN_BARS=5,  DIST_K=0.20, **COMMON),
}

# fallback if no specific profile exists
FALLBACK = dict(ADX_MIN=22, ATR_QUANTILE=0.25, COOLDOWN_BARS=8, DIST_K=0.20, **COMMON)

def generate_signals(df: pd.DataFrame) -> pd.Series:
    sym = os.getenv("BOT_SYMBOL", "EURUSD").upper()
    tf  = os.getenv("BOT_TIMEFRAME", "M15").upper()
    P = PROFILES.get((sym, tf), FALLBACK)

    FAST, SLOW, TREND = P["FAST"], P["SLOW"], P["TREND"]
    ADX_MIN, ADX_PERIOD = P["ADX_MIN"], P["ADX_PERIOD"]
    ATR_QUANTILE = P["ATR_QUANTILE"]
    COOLDOWN_BARS = P["COOLDOWN_BARS"]
    TREND_SLOPE_BARS, DIST_K = P["TREND_SLOPE_BARS"], P["DIST_K"]
    SESSION_HOURS, SESSION_OFFSET_HOURS = P["SESSION_HOURS"], P["SESSION_OFFSET_HOURS"]

    close = df["Close"]; high = df["High"]; low = df["Low"]
    ema_fast  = close.ewm(span=FAST, adjust=False).mean()
    ema_slow  = close.ewm(span=SLOW, adjust=False).mean()
    ema_trend = close.ewm(span=TREND, adjust=False).mean()

    cross_up   = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    cross_down = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    # Strength & volatility
    adx_val = adx(high, low, close, ADX_PERIOD)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    atr_thresh = atr.rolling(200, min_periods=100).quantile(ATR_QUANTILE)
    vol_ok = atr > atr_thresh

    # Session filter (convert broker server time -> target)
    if "time" in df.columns:
        t = df["time"]
        if SESSION_OFFSET_HOURS:
            t = t - pd.to_timedelta(SESSION_OFFSET_HOURS, unit="h")
        session_ok = t.dt.hour.isin(SESSION_HOURS)
    else:
        session_ok = pd.Series(True, index=df.index)

    # Extra quality gates
    slope = ema_trend - ema_trend.shift(TREND_SLOPE_BARS)
    long_ok  = cross_up   & (close > ema_trend) & (slope > 0) & (adx_val >= ADX_MIN) & vol_ok & session_ok & ((close - ema_trend) > DIST_K*atr)
    short_ok = cross_down & (close < ema_trend) & (slope < 0) & (adx_val >= ADX_MIN) & vol_ok & session_ok & ((ema_trend - close) > DIST_K*atr)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_ok]  = 1
    sig[short_ok] = -1

    # Cooldown
    if COOLDOWN_BARS > 0:
        recent = sig.ne(0).rolling(COOLDOWN_BARS).max().shift(1).fillna(0).astype(bool)
        sig = sig.where(~recent, 0)

    return sig
