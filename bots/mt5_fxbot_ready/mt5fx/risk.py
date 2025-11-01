import math

def lots_from_risk_pct(symbol_info, equity:float, risk_pct:float, entry:float, sl:float)->float:
    if equity<=0 or risk_pct<=0: return 0.0
    risk_amt=equity*risk_pct; dist=abs(entry-sl)
    if dist<=0: return 0.0
    vpp=symbol_info.trade_tick_value/symbol_info.trade_tick_size if symbol_info.trade_tick_size>0 else 0
    if vpp<=0: return 0.0
    points=dist/symbol_info.point; loss_per_lot=vpp*points
    lots=risk_amt/loss_per_lot if loss_per_lot>0 else 0.0
    step=getattr(symbol_info,'volume_step',0.01) or 0.01
    lots=max(symbol_info.volume_min, min(symbol_info.volume_max, int(lots/step)*step))
    return lots
