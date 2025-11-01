# scripts/why_no_trade.py
import os, pandas as pd
from mt5fx.utils import load_config
from mt5fx.data import load_rates
from user_strategy.my_alpha import (  # import your plugin constants & helpers
    FAST, SLOW, TREND, ADX_PERIOD, ADX_MIN, ATR_PERIOD, ATR_LOOKBACK,
    ATR_QUANTILE, SESSION_HOURS, SESSION_OFFSET_HOURS, TREND_SLOPE_BARS, DIST_K, COOLDOWN_BARS,
    rma, adx
)

def main():
    cfg = load_config("config.yaml")
    sym = cfg["general"]["symbol"]; tf = cfg["general"]["timeframe"]; look = cfg["general"]["live_context_lookback"]
    df = load_rates(sym, tf, count=look)

    # Recreate adapter enrichments that engine does
    df = df.rename(columns={"open":"bid_o","high":"bid_h","low":"bid_l","close":"bid_c"})
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # approximate ask/mid
    point = 0.00001
    if "spread" in df.columns:
        spread_px = df["spread"].astype(float) * point
    else:
        spread_px = 0.00015
    for k in ("o","h","l","c"):
        df[f"ask_{k}"] = df[f"bid_{k}"] + spread_px
        df[f"mid_{k}"] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2.0

    # Build OHLC for plugin
    df["Open"] = df["mid_o"]; df["High"]=df["mid_h"]; df["Low"]=df["mid_l"]; df["Close"]=df["mid_c"]

    close, high, low = df["Close"], df["High"], df["Low"]
    ema_fast  = close.ewm(span=FAST, adjust=False).mean()
    ema_slow  = close.ewm(span=SLOW, adjust=False).mean()
    ema_trend = close.ewm(span=TREND, adjust=False).mean()
    cross_up   = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    cross_down = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    adx_val = adx(high, low, close, ADX_PERIOD)
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_PERIOD)
    atr_thresh = atr.rolling(ATR_LOOKBACK, min_periods=ATR_LOOKBACK//2).quantile(ATR_QUANTILE)
    vol_ok = atr > atr_thresh

    t = df["time"]
    if SESSION_OFFSET_HOURS:
        t = t - pd.to_timedelta(SESSION_OFFSET_HOURS, unit="h")
    session_ok = t.dt.hour.isin(SESSION_HOURS)

    slope = ema_trend - ema_trend.shift(TREND_SLOPE_BARS)
    dist_long  = (close - ema_trend)
    dist_short = (ema_trend - close)

    i = len(df)-1
    row = {
        "time": df.iloc[i]["time"],
        "cross_up": bool(cross_up.iloc[i]), "cross_down": bool(cross_down.iloc[i]),
        "close>trend": bool(close.iloc[i] > ema_trend.iloc[i]),
        "close<trend": bool(close.iloc[i] < ema_trend.iloc[i]),
        "ADX": float(adx_val.iloc[i]), "ADX_MIN": ADX_MIN, "adx_ok": bool(adx_val.iloc[i] >= ADX_MIN),
        "ATR": float(atr.iloc[i]), "ATR_thresh": float(atr_thresh.iloc[i]), "vol_ok": bool(vol_ok.iloc[i]),
        "session_ok": bool(session_ok.iloc[i]),
        "slope": float(slope.iloc[i]),
        "dist_long_ATR": float(dist_long.iloc[i] / max(atr.iloc[i], 1e-9)),
        "dist_short_ATR": float(dist_short.iloc[i] / max(atr.iloc[i], 1e-9)),
        "DIST_K": DIST_K,
        "cooldown": COOLDOWN_BARS
    }
    print("=== last bar filter diagnostics ===")
    for k,v in row.items(): print(f"{k}: {v}")

if __name__ == "__main__":
    main()
