# user_strategy/trend_follow.py
# Trend-following (EMA 50/200 + ADX + pullback to EMA20)
# Signal = +1 (long) / -1 (short) / 0 (flat), evaluated at bar close.

import pandas as pd
import numpy as np

# ---- default params (you can override via profiles or code edits) ----
EMA_FAST = 50
EMA_SLOW = 200
PULLBACK_MA = 20          # pullback-to-mean filter
ADX_PERIOD = 14
ADX_MIN = 20.0            # trend strength threshold
COOLDOWN = 3              # avoid immediate re-entries (bars)

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _true_range(h, l, c):
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def _adx(h, l, c, n=14):
    # Wilder's ADX (simplified)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # normalize columns (your adapters already create bid/ask/mid_*; prefer mid)
    close = df.get("mid_c", df.get("Close"))
    high  = df.get("mid_h", df.get("High"))
    low   = df.get("mid_l", df.get("Low"))

    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)
    ema_pull = _ema(close, PULLBACK_MA)
    adx = _adx(high, low, close, ADX_PERIOD)

    # long regime only if higher-time trend is up: ema_fast > ema_slow
    up_trend = ema_fast > ema_slow
    dn_trend = ema_fast < ema_slow

    # enter on pullback to EMA20 inside trend when ADX strong
    long_pull  = (up_trend)  & (adx > ADX_MIN) & (close <= ema_pull) & (close.shift(1) > ema_pull.shift(1))
    short_pull = (dn_trend)  & (adx > ADX_MIN) & (close >= ema_pull) & (close.shift(1) < ema_pull.shift(1))

    # cooldown to avoid machine-gunning signals
    sig = pd.Series(0, index=df.index, dtype=int)
    raw_long  = long_pull.astype(int)
    raw_short = (-short_pull.astype(int))

    raw = raw_long + raw_short  # +1, 0, -1
    if COOLDOWN > 0:
        last_fire = -1
        out = []
        for i, v in enumerate(raw.values):
            if v != 0 and (i - last_fire) <= COOLDOWN:
                out.append(0)
            else:
                out.append(v)
                if v != 0:
                    last_fire = i
        sig[:] = out
    else:
        sig = raw

    return sig
