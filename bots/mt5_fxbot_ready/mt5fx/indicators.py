import numpy as np

def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def atr(df, n=14):
    high,low,close=df['mid_h'],df['mid_l'],df['mid_c']; prev=close.shift(1)
    tr=np.maximum(high-low, np.maximum((high-prev).abs(), (low-prev).abs()))
    return tr.rolling(n).mean()
