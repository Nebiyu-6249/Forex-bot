import pandas as pd

def generate_signals(df:pd.DataFrame)->pd.Series:
    ma=df['Close'].rolling(20).mean()
    up=(df['Close'].shift(1)<=ma.shift(1)) & (df['Close']>ma)
    dn=(df['Close'].shift(1)>=ma.shift(1)) & (df['Close']<ma)
    sig=pd.Series(0,index=df.index,dtype=int); sig[up]=1; sig[dn]=-1; return sig
