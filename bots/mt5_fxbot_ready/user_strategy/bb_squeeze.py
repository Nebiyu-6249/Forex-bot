# user_strategy/bb_squeeze.py
# Bollinger "Squeeze" strategy (bar-close signals).
# Entry:
#   SQUEEZE(active) AND (close - SMA20) >= 0.25*ATR14 above/below the SMA20
#   Optional trend filter (SMA100 slope > 0 for longs, < 0 for shorts, or close vs SMA100).
#   One-signal-per-squeeze: only the FIRST qualifying bar inside each squeeze run.
# Squeeze:
#   Baseline (H4 example): BandWidthAbs <= 0.0100
#   Adaptive (default): BandWidthAbs <= rolling 20th percentile over 90 bars
# Exits:
#   Provided to the engine via config (custom SL/TP mode) -> SL = prior bar extreme, TP = 2R.
#
# Signal interface REQUIRED by engine/backtester:
#   generate_signals(df: pd.DataFrame) -> pd.Series in {-1,0,1} aligned to df.index

import numpy as np
import pandas as pd

# ---- Tunables (safe defaults for intraday FX; adjust per pair/TF if needed) ----
BB_PERIOD = 20
BB_K = 2.2
ATR_PERIOD = 14

# Squeeze mode
SQUEEZE_MODE = "adaptive"        # "adaptive" (default) or "baseline"
BASELINE_BW_MAX = 0.0100         # 100 pips typical for H4 examples in PDFs
ADAPT_LOOKBACK = 90
ADAPT_PERCENTILE = 0.20

# Entry buffer to avoid barely-across closes
ATR_BUFFER_K = 0.3              # require |close - SMA20| >= 0.25*ATR

# Trend filter
USE_TREND_FILTER = True
TREND_MA = 100
TREND_SLOPE_BARS = 3             # small slope lookback
TREND_RULE = "slope_or_price"    # "slope_only" | "price_only" | "slope_or_price"

# Session filter (engine also has spread/circuit breakers; this is an extra timing quality gate)
SESSION_FILTER = True
SESSION_HOURS = set(range(7, 20))     # 07:00–19:59 (TARGET clock)
SESSION_OFFSET_HOURS = +2             # broker server time -> TARGET (your broker GMT+2)

# One-signal-per-squeeze:
ONE_PER_SQUEEZE = True

# Optional cooldown
COOLDOWN_BARS = 0

def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0/float(n), adjust=False).mean()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return _rma(tr, n)

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Normalize OHLC (the adapters already create these; just be defensive)
    close = df["Close"] if "Close" in df.columns else df["mid_c"]
    high  = df["High"]  if "High"  in df.columns else df["mid_h"]
    low   = df["Low"]   if "Low"   in df.columns else df["mid_l"]

    # Bollinger bands
    sma20 = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    std20 = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper = sma20 + BB_K*std20
    lower = sma20 - BB_K*std20
    bw_abs = upper - lower

    # ATR for buffers/robustness
    atr14 = _atr(high, low, close, ATR_PERIOD)

    # Squeeze definition
    if SQUEEZE_MODE.lower() == "baseline":
        squeeze = bw_abs <= BASELINE_BW_MAX
    else:
        thr = bw_abs.rolling(ADAPT_LOOKBACK, min_periods=int(ADAPT_LOOKBACK*0.8)).quantile(ADAPT_PERCENTILE)
        squeeze = bw_abs <= thr

    # Core directional conditions with ATR buffer
    above_mid = (close > sma20) & ((close - sma20) >= ATR_BUFFER_K*atr14)
    below_mid = (close < sma20) & ((sma20 - close) >= ATR_BUFFER_K*atr14)

    # Trend filter
    if USE_TREND_FILTER:
        sma100 = close.rolling(TREND_MA, min_periods=TREND_MA).mean()
        slope  = sma100 - sma100.shift(TREND_SLOPE_BARS)
        if TREND_RULE == "slope_only":
            long_trend_ok  = slope > 0
            short_trend_ok = slope < 0
        elif TREND_RULE == "price_only":
            long_trend_ok  = close > sma100
            short_trend_ok = close < sma100
        else:  # slope_or_price
            long_trend_ok  = (slope > 0) | (close > sma100)
            short_trend_ok = (slope < 0) | (close < sma100)
    else:
        long_trend_ok = short_trend_ok = pd.Series(True, index=df.index)

    long_raw  = squeeze & above_mid & long_trend_ok
    short_raw = squeeze & below_mid & short_trend_ok

    # One-signal-per-squeeze run
    if ONE_PER_SQUEEZE:
        run_id = (squeeze != squeeze.shift(1)).cumsum().fillna(0).astype(int)
        long_ord  = long_raw.groupby(run_id).cumsum()
        short_ord = short_raw.groupby(run_id).cumsum()
        long_ok   = long_raw  & (long_ord  == 1)
        short_ok  = short_raw & (short_ord == 1)
    else:
        long_ok, short_ok = long_raw, short_raw

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_ok]  = 1
    sig[short_ok] = -1

    # Optional session filter: align server -> target hours
    if SESSION_FILTER and "time" in df.columns:
        t = df["time"] - pd.to_timedelta(SESSION_OFFSET_HOURS, unit="h")
        sig = sig.where(t.dt.hour.isin(SESSION_HOURS), 0)

    # Optional cooldown after any signal
    if COOLDOWN_BARS > 0:
        mask_recent = sig.ne(0).rolling(COOLDOWN_BARS).max().shift(1).fillna(0).astype(bool)
        sig = sig.where(~mask_recent, 0)

    return sig
