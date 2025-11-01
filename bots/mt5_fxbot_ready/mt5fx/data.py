import pandas as pd, MetaTrader5 as mt5
from .mt5_client import MT5

def load_rates(symbol:str, timeframe:str, count:int=1000)->pd.DataFrame:
    cli=MT5(); cli.init()
    try:
        info=cli.ensure_symbol(symbol) 
        symbol = getattr(cli, "last_symbol", symbol)
        tf=cli.timeframe(timeframe)
        rates=mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates)==0: raise RuntimeError('No rates')
        df=pd.DataFrame(rates)
        df.rename(columns={'open':'bid_o','high':'bid_h','low':'bid_l','close':'bid_c'}, inplace=True)
        df['time']=pd.to_datetime(df['time'], unit='s'); df['complete']=True
        point=info.point; df['spread_points']=df['spread'].astype(float); df['spread_px']=df['spread_points']*point
        for k in ('o','h','l','c'):
            df[f'ask_{k}']=df[f'bid_{k}']+df['spread_px']; df[f'mid_{k}']=(df[f'bid_{k}']+df[f'ask_{k}'])/2.0
        cols=['time','complete','bid_o','bid_h','bid_l','bid_c','ask_o','ask_h','ask_l','ask_c','mid_o','mid_h','mid_l','mid_c']
        return df[cols]
    finally:
        cli.shutdown()
