# user_strategy/mean_revert.py
# Mean reversion with re-entry confirmation and adaptive filters

import pandas as pd
import numpy as np

# ---- Tunables (safe defaults) ----
BB_N = 20
INNER_K = 2.2          # inner band for re-entry
OUTER_K = 3.0          # outer band for "extreme" test (prev bar)
ADX_N = 14
ADX_MAX = 18.0         # looser than 10 to allow more trades
SLOPE_ATR_MAX = 2.0e-05  # cap on SMA20 slope (in ATR units)
RSI_N = 2
RSI_LONG_STRICT = 6.0
RSI_SHORT_STRICT = 94.0
RSI_LONG_LOOSE = 12.0
RSI_SHORT_LOOSE = 88.0
ONE_PER_BAR = True
COOLDOWN = 4

def _sma(s, n): return s.rolling(n, min_periods=n).mean()

def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def _atr(h, l, c, n=14):
    return _true_range(h, l, c).ewm(alpha=1/n, adjust=False).mean()

def _adx(h, l, c, n=14):
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(h, l, c, n)
    plus_di  = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df.get("mid_c", df.get("Close"))
    high  = df.get("mid_h", df.get("High"))
    low   = df.get("mid_l", df.get("Low"))

    sma = _sma(close, BB_N)
    std = close.rolling(BB_N, min_periods=BB_N).std(ddof=0)

    upper_in  = sma + INNER_K * std
    lower_in  = sma - INNER_K * std
    upper_out = sma + OUTER_K * std
    lower_out = sma - OUTER_K * std

    adx  = _adx(high, low, close, ADX_N)
    rsi2 = _rsi(close, RSI_N)
    atr  = _atr(high, low, close, 14)
    slope = sma.diff().abs() / atr

    # Low-trend regime: flat-ish + low ADX
    low_trend = (adx <= ADX_MAX) & (slope <= SLOPE_ATR_MAX)

    pc = close.shift(1)

    # --- Re-entry confirmation ---
    # strict: prev close beyond outer band, now back inside inner band, RSI extreme + reversal candle
    long_strict  = low_trend & (pc < lower_out.shift(1)) & (close >= lower_in) & (rsi2 <= RSI_LONG_STRICT) & (close > pc)
    short_strict = low_trend & (pc > upper_out.shift(1)) & (close <= upper_in) & (rsi2 >= RSI_SHORT_STRICT) & (close < pc)

    # loose: prev close beyond inner band (but not necessarily outer), now back inside inner band + looser RSI
    long_loose  = low_trend & (pc < lower_in.shift(1)) & (close >= lower_in) & (rsi2 <= RSI_LONG_LOOSE) & (close > pc)
    short_loose = low_trend & (pc > upper_in.shift(1)) & (close <= upper_in) & (rsi2 >= RSI_SHORT_LOOSE) & (close < pc)

    long_raw  = long_strict | long_loose
    short_raw = short_strict | short_loose

    raw = pd.Series(0, index=df.index, dtype=int)
    raw[long_raw] = 1
    raw[short_raw] = -1

    # --- Cooldown to avoid ping-pong ---
    if COOLDOWN > 0:
        last_fire = -10_000
        cooled = []
        for i, v in enumerate(raw.values):
            if v != 0 and (i - last_fire) <= COOLDOWN:
                cooled.append(0)
            else:
                cooled.append(v)
                if v != 0:
                    last_fire = i
        sig = pd.Series(cooled, index=raw.index, dtype=int)
    else:
        sig = raw

    return sig
