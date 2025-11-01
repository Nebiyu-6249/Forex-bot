from pathlib import Path
from .utils import jload, jdump

class State:
    def __init__(self, out_dir:str, symbol:str, timeframe:str):
        safe=f"{symbol}_{timeframe}".replace('.','_')
        self.path = Path(out_dir)/'state'/f'state_{safe}.json'
        self.data = jload(self.path, default={'last_complete_time': None})
    def get_last_time(self):
        return self.data.get('last_complete_time')
    def set_last_time(self, ts:str):
        self.data['last_complete_time']=ts; jdump(self.data, self.path)
