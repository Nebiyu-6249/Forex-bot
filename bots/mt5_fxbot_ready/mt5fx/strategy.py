import pandas as pd
from .indicators import ema, atr

def ema_cross_with_trend(df:pd.DataFrame, fast=20, slow=50, trend=200):
    out=df.copy(); out['ema_fast']=ema(out['mid_c'], fast); out['ema_slow']=ema(out['mid_c'], slow); out['ema_trend']=ema(out['mid_c'], trend)
    cu=(out['ema_fast'].shift(1)<=out['ema_slow'].shift(1)) & (out['ema_fast']>out['ema_slow'])
    cd=(out['ema_fast'].shift(1)>=out['ema_slow'].shift(1)) & (out['ema_fast']<out['ema_slow'])
    long_ok=cu & (out['mid_c']>out['ema_trend']); short_ok=cd & (out['mid_c']<out['ema_trend'])
    sig=pd.Series(0,index=out.index,dtype=int); sig[long_ok]=1; sig[short_ok]=-1
    return sig.shift(1).fillna(0).astype(int)

def attach_atr(df:pd.DataFrame, n=14):
    out=df.copy(); h,l,c=out['mid_h'],out['mid_l'],out['mid_c']; pc=c.shift(1)
    tr_hl=h-l; tr_hc=(h-pc).abs(); tr_lc=(l-pc).abs()
    out['ATR']=pd.concat([tr_hl,tr_hc,tr_lc],axis=1).max(axis=1).rolling(n).mean(); return out
