# mt5fx/adapters.py
import pandas as pd
import numpy as np

def enrich_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure standard OHLC + Bid/Ask columns exist so user plugins can rely on them.
    - Map from mid_* to Open/High/Low/Close if needed
    - Provide Bid/Ask shortcuts from bid_c/ask_c if present
    - Preserve df['time'] as pandas datetime if available
    """
    out = df.copy()

    mapping = {
        "Open": "mid_o",
        "High": "mid_h",
        "Low":  "mid_l",
        "Close":"mid_c",
        # also accept lowercase aliases if present
        "open": "mid_o",
        "high": "mid_h",
        "low":  "mid_l",
        "close":"mid_c",
        "Mid":  "mid_c",
        "mid":  "mid_c",
    }

    for dst, src in mapping.items():
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]

    # Provide Bid/Ask helpers if missing
    if ("Bid" not in out.columns) and ("bid_c" in out.columns):
        out["Bid"] = out["bid_c"]
    if ("Ask" not in out.columns) and ("ask_c" in out.columns):
        out["Ask"] = out["ask_c"]

    # Make sure time is datetime64 if present
    if "time" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["time"]):
        try:
            out["time"] = pd.to_datetime(out["time"])
        except Exception:
            pass

    return out
