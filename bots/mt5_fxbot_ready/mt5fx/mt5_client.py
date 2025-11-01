import MetaTrader5 as mt5
from .utils import load_env, env

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

class MT5:
    def __init__(self):
        load_env()
        self.path = env("MT5_PATH", required=True)
        self.login = env("MT5_LOGIN", required=True, cast=int)
        self.password = env("MT5_PASSWORD", required=True)
        self.server = env("MT5_SERVER", required=True)
        self.initialized = False
        self.last_symbol = None  # resolved broker-specific symbol

    def init(self):
        if not mt5.initialize(self.path):
            raise RuntimeError(f"initialize() failed, error code: {mt5.last_error()}")
        if not mt5.login(self.login, password=self.password, server=self.server):
            raise RuntimeError(f"login() failed, error code: {mt5.last_error()}")
        self.initialized = True

    def shutdown(self):
        if self.initialized:
            mt5.shutdown()
            self.initialized = False

    def _resolve_symbol_name(self, symbol: str) -> str:
        """Return the broker-specific symbol (handles suffixes like .r, m, micro, etc.)."""
        base = symbol.replace("/", "").upper()
        # exact symbol exists?
        if mt5.symbol_info(base) is not None:
            return base
        # try prefix match (EURUSD*, XAUUSD*, etc.)
        pref = mt5.symbols_get(f"{base}*") or []
        if pref:
            # prefer names that start with base and are shortest
            pref = sorted((s.name for s in pref), key=lambda n: (0 if n.upper()==base else 1, len(n)))
            return pref[0]
        # try contains match (*EURUSD*)
        cont = mt5.symbols_get(f"*{base}*") or []
        if cont:
            cont = sorted((s.name for s in cont), key=lambda n: (0 if n.upper()==base else 1 if n.upper().startswith(base) else 2, len(n)))
            return cont[0]
        # give up; let caller raise with original name
        return symbol

    def ensure_symbol(self, symbol: str):
        resolved = self._resolve_symbol_name(symbol)
        info = mt5.symbol_info(resolved)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} not found (tried: {resolved})")
        if not info.visible:
            if not mt5.symbol_select(resolved, True):
                raise RuntimeError(f"symbol_select({resolved}) failed")
        self.last_symbol = resolved
        return mt5.symbol_info(resolved)

    def timeframe(self, tf_str: str):
        if tf_str not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {tf_str}")
        return TIMEFRAME_MAP[tf_str]
