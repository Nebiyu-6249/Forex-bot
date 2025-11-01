import importlib, pandas as pd
from typing import Optional
from .adapters import enrich_ohlc

def _to_signal_series(obj, df:pd.DataFrame)->Optional[pd.Series]:
    if obj is None: return None
    if isinstance(obj, pd.Series): return obj.astype(int)
    if isinstance(obj, pd.DataFrame):
        if 'signal' in obj.columns: return obj['signal'].astype(int)
        cols={c.lower():c for c in obj.columns}
        if 'long' in cols and 'short' in cols:
            longs=obj[cols['long']].astype(bool).astype(int)
            shorts=obj[cols['short']].astype(bool).astype(int)
            return (longs - shorts).astype(int)
    return None

def compute_plugin_signal(df:pd.DataFrame, module_path:str, func_name:str, shift_next_bar:bool=True)->pd.Series:
    # Load user's module and compute a signal Series in {-1,0,1}
    m=importlib.import_module(module_path)
    fn=getattr(m, func_name, None)
    if fn is None:
        for cand in ['generate_signals','strategy','signals','build_signals','run','make_signals']:
            fn=getattr(m, cand, None)
            if fn: break
    if fn is None:
        raise RuntimeError(f'Could not find function {func_name} in module {module_path}')
    dfe=enrich_ohlc(df); result=fn(dfe); sig=_to_signal_series(result, dfe)
    if sig is None:
        raise RuntimeError("Plugin must return Series ints or DataFrame with 'signal' or 'long'/'short' columns")
    sig=sig.reindex(df.index).fillna(0).astype(int)
    return sig.shift(1).fillna(0).astype(int) if shift_next_bar else sig
