# filename: mt5fx/engine.py
import os
import math
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5

from .mt5_client import MT5
from .strategy import ema_cross_with_trend, attach_atr
from .strategy_loader import compute_plugin_signal
from .state import State
from .logging_utils import setup_logging
from . import advisor as advisor_mod
from .regime_detector import detect_regime, Regime
from .multi_tf import should_block_signal
from .journal import TradeJournal


def _tf_minutes(tf_str: str) -> int:
    """Convert timeframe string (e.g., 'M15','H1','D1') to minutes."""
    tf_str = tf_str.upper()
    if tf_str.startswith("M"):
        return int(tf_str[1:])
    if tf_str.startswith("H"):
        return 60 * int(tf_str[1:])
    if tf_str == "D1":
        return 1440
    if tf_str == "W1":
        return 10080
    if tf_str == "MN1":
        return 43200
    raise ValueError(f"Unsupported timeframe: {tf_str}")


# Map timeframe strings to MT5 enums (for multi-tf lookups)
_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2,
    "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
}


class LiveEngine:
    """
    Live trading engine (MT5):
      - Spread-aware entries (Ask for long, Bid for short)
      - Uses either built-in EMA strategy or plugin strategy (+1/-1/0 on close; execute next bar)
      - Exits management: breakeven & ATR trailing & optional time-stop
      - Optional scale-out at mid-band (partial close) with TP retarget for remainder
      - Guards: spread filter, circuit breakers (daily loss, max trades, consecutive losses, Friday cutoff)
      - Regime detection: auto-classifies TRENDING/RANGING/CHOPPY
      - Multi-timeframe confirmation: blocks counter-trend signals
      - Dynamic position sizing with drawdown reduction
      - Health heartbeat file
      - Trade journal (SQLite)
      - MT5 reconnection with exponential backoff
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.log = setup_logging(cfg["general"]["out_dir"])

        # connect to MT5 terminal using .env values
        self.mt5 = MT5()
        self.mt5.init()

        self.symbol = cfg["general"]["symbol"]
        self.info = self.mt5.ensure_symbol(self.symbol)
        # use resolved broker-specific symbol (e.g., EURUSD.r)
        self.symbol = getattr(self.mt5, "last_symbol", self.symbol)

        self.tf_str = cfg["general"]["timeframe"]
        self.tf = self.mt5.timeframe(self.tf_str)
        self.tf_min = _tf_minutes(self.tf_str)
        self.lookback = cfg["general"]["live_context_lookback"]

        self.state = State(cfg["general"]["out_dir"], self.symbol, self.tf_str)

        # runtime flags/ caches
        self._be_applied = set()   # tickets that already had breakeven applied
        self._scaled_once = set()  # tickets already scaled out
        self._last_df = None       # last fetched bars (for scale-out mid-band calc)
        self._peak_equity = self._account_equity() or 10000.0
        self._reconnect_backoff = 2  # seconds

        # Trade journal
        out_dir = cfg["general"]["out_dir"]
        self.journal = TradeJournal(os.path.join(out_dir, "trade_journal.db"))

        # Heartbeat file path
        self._heartbeat_path = os.path.join(out_dir, "heartbeat.txt")

    def __del__(self):
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    # ---------------------- Helpers ----------------------

    def _write_heartbeat(self):
        """Write heartbeat file for health monitoring."""
        try:
            Path(self._heartbeat_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._heartbeat_path, "w") as f:
                f.write(f"{datetime.utcnow().isoformat()}\n")
                f.write(f"symbol={self.symbol}\n")
                f.write(f"timeframe={self.tf_str}\n")
                equity = self._account_equity()
                f.write(f"equity={equity:.2f}\n")
                pos = self._get_open_position()
                f.write(f"open_position={'yes' if pos else 'no'}\n")
        except Exception:
            pass

    def _try_reconnect(self):
        """Attempt to reconnect to MT5 with exponential backoff."""
        max_backoff = 60
        for attempt in range(5):
            wait = min(self._reconnect_backoff * (2 ** attempt), max_backoff)
            self.log.warning("MT5 reconnection attempt %d, waiting %ds...", attempt + 1, wait)
            time.sleep(wait)
            try:
                mt5.shutdown()
                self.mt5.init()
                self.info = self.mt5.ensure_symbol(self.symbol)
                self.symbol = getattr(self.mt5, "last_symbol", self.symbol)
                self.log.info("MT5 reconnected successfully.")
                return True
            except Exception as e:
                self.log.error("Reconnection attempt %d failed: %s", attempt + 1, e)
        self.log.critical("Failed to reconnect to MT5 after 5 attempts.")
        return False

    def _has_open_position(self) -> bool:
        poss = mt5.positions_get(symbol=self.symbol)
        return (poss is not None) and (len(poss) > 0)

    def _get_open_position(self):
        poss = mt5.positions_get(symbol=self.symbol)
        if poss and len(poss) > 0:
            return poss[0]
        return None

    def _account_equity(self) -> float:
        a = mt5.account_info()
        return float(a.equity) if a else 0.0

    def _compute_signal(self, df: pd.DataFrame) -> pd.Series:
        s = self.cfg["strategy"]
        os.environ["BOT_SYMBOL"] = self.symbol
        os.environ["BOT_TIMEFRAME"] = self.tf_str
        if s.get("use_plugin", False):
            return compute_plugin_signal(
                df,
                s["plugin_module"],
                s.get("plugin_function", "generate_signals"),
                s.get("plugin_shift_next_bar", True),
            )
        return ema_cross_with_trend(df, fast=s["fast_ema"], slow=s["slow_ema"], trend=s["trend_ema"])

    def _place_market(self, side: str, sl: float, tp: float, deviation_points: int = 10, lots: float = 0.1):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.log.error("No tick data for %s", self.symbol)
            return None

        price = tick.ask if side == "long" else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation_points,
            "magic": 20251019,
            "comment": f"mt5fx {self.tf_str}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        # retry with FOK if broker refuses IOC
        if result and result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode in (mt5.TRADE_RETCODE_INVALID_FILL, mt5.TRADE_RETCODE_NO_CHANGES):
                req["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(req)
        return result

    def _spread_points(self) -> int:
        """Current spread in points (integer)."""
        t = mt5.symbol_info_tick(self.symbol)
        if not t:
            return 999999
        return int(round((t.ask - t.bid) / self.info.point))

    def _round_lots(self, lots: float) -> float:
        si = mt5.symbol_info(self.symbol)
        if not si:
            return round(max(0.01, lots), 2)
        step = float(si.volume_step or 0.01)
        vmin = float(si.volume_min or step)
        vmax = float(si.volume_max or 100.0)
        lots = max(vmin, min(vmax, (lots // step) * step))
        prec = 0 if step >= 1 else max(0, int(round(-math.log10(step))))
        return round(lots, prec)

    def _modify_sl_tp(self, ticket, new_sl=None, new_tp=None):
        pos = next((p for p in (mt5.positions_get(symbol=self.symbol) or []) if p.ticket == ticket), None)
        if not pos:
            return None
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": float(new_sl if new_sl is not None else pos.sl),
            "tp": float(new_tp if new_tp is not None else pos.tp),
        }
        return mt5.order_send(req)

    def _lots_from_risk_pct(self, symbol: str, equity: float, risk_pct: float,
                            entry: float, sl: float) -> float:
        """
        Position size (lots) from equity risk. Uses MT5 symbol specs.
        Includes dynamic sizing: reduces risk during drawdowns.
        """
        si = mt5.symbol_info(symbol)
        if not si:
            return 0.0

        # Dynamic sizing: reduce risk during drawdowns
        r = self.cfg.get("risk", {})
        if r.get("dynamic_sizing", False):
            drawdown_pct = 1.0 - (equity / self._peak_equity) if self._peak_equity > 0 else 0.0
            reduce_at = float(r.get("drawdown_reduce_at_pct", 0.05))
            halt_at = float(r.get("drawdown_halt_at_pct", 0.10))

            if drawdown_pct >= halt_at:
                self.log.warning("Drawdown %.1f%% >= halt threshold %.1f%%. No new trades.", drawdown_pct * 100, halt_at * 100)
                return 0.0
            elif drawdown_pct >= reduce_at:
                risk_pct *= 0.5  # half risk during drawdown
                self.log.info("Drawdown %.1f%% — reducing risk to %.2f%%", drawdown_pct * 100, risk_pct * 100)

        tick_value = float(si.trade_tick_value)
        tick_size = float(si.trade_tick_size or si.point)
        vmin = float(si.volume_min)
        vstep = float(si.volume_step or 0.01)
        vmax = float(si.volume_max)

        # Cap by config max_lot
        config_max = float(r.get("max_lot", vmax))
        vmax = min(vmax, config_max)

        stop_dist = abs(float(entry) - float(sl))
        if stop_dist <= 0 or tick_value <= 0 or tick_size <= 0:
            return 0.0

        ticks_to_sl = stop_dist / tick_size
        loss_per_lot = ticks_to_sl * tick_value
        risk_amount = max(0.0, float(equity) * float(risk_pct))
        raw_lots = 0.0 if loss_per_lot <= 0 else (risk_amount / loss_per_lot)

        lots = max(vmin, min(vmax, (raw_lots // vstep) * vstep))
        prec = max(0, int(round(-math.log10(vstep)))) if vstep < 1 else 0
        return round(lots, prec)

    def _seconds_to_next_bar(self) -> int:
        """Calculate seconds until the next bar close for adaptive polling."""
        now = datetime.utcnow()
        minutes_since_midnight = now.hour * 60 + now.minute
        current_bar_start = (minutes_since_midnight // self.tf_min) * self.tf_min
        next_bar_start = current_bar_start + self.tf_min
        next_bar_time_minutes = next_bar_start - minutes_since_midnight
        seconds = max(5, next_bar_time_minutes * 60 - now.second)
        # Cap at 5 minutes to avoid sleeping too long
        return min(seconds, 300)

    # ---------------------- Scale-out (LIVE) ----------------------

    def _maybe_scale_out(self, pos, last_row):
        """
        If enabled, close scale_out_fraction at the mid-band the first time it's touched,
        then move TP of the remainder to rr_after_scale * R.
        """
        ex = self.cfg.get("exits", {})
        if not ex.get("scale_out_enable", False):
            return
        if pos.ticket in self._scaled_once:
            return
        df = self._last_df
        if df is None or len(df) < max(25, int(ex.get("bb_period_for_mean", 20)) + 1):
            return

        if "mid_c" in df.columns:
            series = df["mid_c"]
        else:
            series = (df["bid_c"] + df["ask_c"]) / 2.0
        n = int(ex.get("bb_period_for_mean", 20))
        mid = series.rolling(n, min_periods=n).mean().iloc[-1]

        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"

        bid_hi = float(df.iloc[-1]["bid_h"])
        ask_lo = float(df.iloc[-1]["ask_l"])
        touched = (side == "long" and bid_hi >= mid) or (side == "short" and ask_lo <= mid)
        if not touched:
            return

        frac = float(ex.get("scale_out_fraction", 0.50))
        close_vol = self._round_lots(float(pos.volume) * frac)
        if close_vol <= 0:
            return

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return
        price = tick.bid if side == "long" else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "position": pos.ticket,
            "volume": close_vol,
            "type": mt5.ORDER_TYPE_SELL if side == "long" else mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 10,
            "magic": 20251021,
            "comment": "scale_out",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)

        rr = float(ex.get("rr_after_scale", 3.0))
        entry = float(pos.price_open)
        sl = float(pos.sl)
        risk = (entry - sl) if side == "long" else (sl - entry)
        if risk > 0:
            new_tp = entry + rr * risk if side == "long" else entry - rr * risk
            self._modify_sl_tp(pos.ticket, new_sl=None, new_tp=new_tp)

        self._scaled_once.add(pos.ticket)
        self.log.info("Scaled out %.2f lots at mid; moved TP to %.5f (%.2fR).", close_vol, (new_tp if risk > 0 else float('nan')), rr)

    # ---------------------- Position management ----------------------

    def _manage_open_position(self, last_row, exits_cfg):
        pos = self._get_open_position()
        if pos is None:
            return

        self._maybe_scale_out(pos, last_row)

        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
        entry = float(pos.price_open)
        sl = float(pos.sl)

        atr = float(last_row["ATR"]) if "ATR" in last_row and not pd.isna(last_row["ATR"]) else None
        if atr is None or atr <= 0:
            return

        be_on = bool(exits_cfg.get("breakeven_enable", False))
        be_R = float(exits_cfg.get("breakeven_at_R", 0.6))
        trail_on = bool(exits_cfg.get("trail_enable", False))
        trail_k = float(exits_cfg.get("trail_atr_mult", 1.0))
        max_hold = int(exits_cfg.get("max_hold_bars", 0))

        bars_open = 0
        try:
            if getattr(pos, "time", None):
                minutes = (datetime.utcnow() - datetime.utcfromtimestamp(int(pos.time))).total_seconds() / 60.0
                bars_open = int(minutes // self.tf_min)
        except Exception:
            pass

        changed = False
        risk_R = (entry - sl) if side == "long" else (sl - entry)

        # 1) Breakeven
        if be_on and (pos.ticket not in self._be_applied) and risk_R > 0:
            be_level = entry + be_R * risk_R if side == "long" else entry - be_R * risk_R
            price_c = float(last_row["bid_c"] if side == "long" else last_row["ask_c"])
            if (side == "long" and price_c >= be_level) or (side == "short" and price_c <= be_level):
                if (side == "long" and entry > sl) or (side == "short" and entry < sl):
                    self._modify_sl_tp(pos.ticket, new_sl=entry, new_tp=None)
                    self._be_applied.add(pos.ticket)
                    changed = True

        # 2) ATR trailing
        if trail_on:
            if side == "long":
                cand = float(last_row["bid_c"]) - trail_k * atr
                if cand > sl:
                    self._modify_sl_tp(pos.ticket, new_sl=cand, new_tp=None)
                    changed = True
            else:
                cand = float(last_row["ask_c"]) + trail_k * atr
                if cand < sl:
                    self._modify_sl_tp(pos.ticket, new_sl=cand, new_tp=None)
                    changed = True

        # 3) Time stop
        if max_hold and bars_open >= max_hold:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                px = tick.bid if side == "long" else tick.ask
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": float(pos.volume),
                    "type": mt5.ORDER_TYPE_SELL if side == "long" else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": px,
                    "deviation": 10,
                    "magic": 20251019,
                    "comment": f"mt5fx time-stop {self.tf_str}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(req)
                self.log.info("Closed position %s due to time-stop (%s bars).", pos.ticket, bars_open)
                return

        if changed:
            self.log.info("Modified SL/TP for ticket %s", pos.ticket)

    # ---------------------- Circuit breakers / filters ----------------------

    def _check_circuit_breakers(self, df, filters_cfg, breakers_cfg) -> tuple[bool, str]:
        # Spread filter
        max_spread = int(filters_cfg.get("max_spread_points", 999999))
        sp = self._spread_points()
        if sp > max_spread:
            return True, f"spread {sp} > limit {max_spread}"

        if not breakers_cfg.get("enabled", False):
            return False, ""

        now = datetime.utcnow()
        start = datetime(now.year, now.month, now.day)
        deals = mt5.history_deals_get(start, now) or []
        sym_deals = [d for d in deals if getattr(d, "symbol", "") == self.symbol]
        pnl_today = sum(float(getattr(d, "profit", 0.0)) for d in sym_deals)

        pos_ids = set(getattr(d, "position_id", getattr(d, "position", 0)) for d in sym_deals if getattr(d, "position_id", None) or getattr(d, "position", None))
        trades_today = len(pos_ids)

        max_loss = float(breakers_cfg.get("max_daily_loss_currency", 0.0))
        if max_loss > 0 and pnl_today <= -abs(max_loss):
            return True, f"daily loss {pnl_today:.2f} <= -{abs(max_loss):.2f}"

        max_trades = int(breakers_cfg.get("max_trades_per_day", 999999))
        if trades_today >= max_trades:
            return True, f"trades per day {trades_today} >= {max_trades}"

        # Consecutive losses
        max_consec_losses = int(breakers_cfg.get("max_consec_losses", 0))
        if max_consec_losses > 0 and sym_deals:
            sym_deals.sort(key=lambda d: getattr(d, "time", 0), reverse=True)
            consec = 0
            seen_positions = set()
            for d in sym_deals:
                pid = getattr(d, "position_id", getattr(d, "position", None))
                if not pid or pid in seen_positions:
                    continue
                seen_positions.add(pid)
                pr = float(getattr(d, "profit", 0.0))
                if pr < 0:
                    consec += 1
                    if consec >= max_consec_losses:
                        return True, f"consecutive losses {consec} >= {max_consec_losses}"
                else:
                    break

        # Friday cutoff
        cutoff = breakers_cfg.get("no_trade_friday_after_hour", None)
        if cutoff is not None:
            last_time = df.iloc[-1]["time"]
            if int(last_time.weekday()) == 4 and int(last_time.hour) >= int(cutoff):
                return True, "Friday cutoff hour reached"

        return False, ""

    def _check_drawdown_breaker(self) -> tuple[bool, str]:
        """Check max drawdown from peak equity."""
        r = self.cfg.get("risk", {})
        halt_at = float(r.get("drawdown_halt_at_pct", 0.10))

        equity = self._account_equity()
        if equity <= 0:
            return False, ""

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown_pct = 1.0 - (equity / self._peak_equity)
        if drawdown_pct >= halt_at:
            return True, f"drawdown {drawdown_pct*100:.1f}% >= halt {halt_at*100:.1f}%"

        return False, ""

    # ---------------------- Regime & Multi-TF ----------------------

    def _check_regime(self, df) -> Regime:
        """Detect current market regime."""
        regime_cfg = self.cfg.get("regime", {})
        if not regime_cfg.get("enabled", False):
            return Regime.RANGING  # default

        return detect_regime(
            df,
            adx_trending_threshold=float(regime_cfg.get("adx_trending_threshold", 25)),
            adx_choppy_threshold=float(regime_cfg.get("adx_choppy_threshold", 15)),
            bb_width_ranging_pctl=float(regime_cfg.get("bb_width_ranging_pctl", 30)),
            lookback_bars=int(regime_cfg.get("lookback_bars", 100)),
        )

    def _check_mtf_block(self, signal: int) -> tuple[bool, str]:
        """Check if multi-timeframe analysis blocks the signal."""
        mtf_cfg = self.cfg.get("multi_timeframe", {})
        if not mtf_cfg.get("enabled", False):
            return False, ""

        conf_tf_str = mtf_cfg.get("confirmation_tf", "H1")
        trend_tf_str = mtf_cfg.get("trend_tf", "H4")
        conf_tf = _TF_MAP.get(conf_tf_str, mt5.TIMEFRAME_H1)
        trend_tf = _TF_MAP.get(trend_tf_str, mt5.TIMEFRAME_H4)

        return should_block_signal(
            signal=signal,
            symbol=self.symbol,
            confirmation_tf=conf_tf,
            trend_tf=trend_tf,
            ema_fast=int(mtf_cfg.get("ema_fast", 50)),
            ema_slow=int(mtf_cfg.get("ema_slow", 200)),
            block_counter_trend=bool(mtf_cfg.get("block_counter_trend", True)),
        )

    # ---------------------- Main loop ----------------------

    def run(self):
        self.log.info("Live loop started for %s @ %s", self.symbol, self.tf_str)
        last_seen = self.state.get_last_time()

        exits_cfg = self.cfg.get("exits", {})
        filters_cfg = self.cfg.get("filters", {})
        breakers_cfg = self.cfg.get("circuit_breakers", {})
        advisor_enabled = bool(self.cfg.get("advisor", {}).get("enable", False))
        debug_cfg = self.cfg.get("debug", {})

        while True:
            try:
                # Write heartbeat
                self._write_heartbeat()

                # pull recent bars
                rates = mt5.copy_rates_from_pos(self.symbol, self.tf, 0, self.lookback)
                if rates is None or len(rates) == 0:
                    self.log.warning("No rate data received. Attempting reconnect...")
                    if not self._try_reconnect():
                        time.sleep(60)
                    continue

                df = pd.DataFrame(rates)
                df.rename(columns={"open": "bid_o", "high": "bid_h", "low": "bid_l", "close": "bid_c"}, inplace=True)
                df["time"] = pd.to_datetime(df["time"], unit="s")

                # approximate ask/mid using per-bar spread * point
                spread_px = (df["spread"].astype(float) * self.info.point) if "spread" in df.columns else (self.info.spread * self.info.point)
                for k in ("o", "h", "l", "c"):
                    df[f"ask_{k}"] = df[f"bid_{k}"] + spread_px
                    df[f"mid_{k}"] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2.0

                self._last_df = df

                # act on *new completed bar* only
                last_complete_iso = df.iloc[-1]["time"].isoformat()
                if last_seen is None or last_complete_iso > last_seen:
                    s = self.cfg["strategy"]
                    r = self.cfg["risk"]

                    # compute signal & ATR
                    sig = self._compute_signal(df)
                    df = attach_atr(df, n=s["atr_period"])
                    last = df.iloc[-1]
                    last_sig = int(sig.iloc[-1])

                    # manage any existing position every bar
                    self._manage_open_position(last, exits_cfg)

                    # --- SIGNAL HANDLING + ORDER PLACEMENT ---
                    if last_sig != 0 and not pd.isna(last["ATR"]) and last["ATR"] > 0:
                        if not self._has_open_position():

                            # Drawdown breaker
                            dd_block, dd_reason = self._check_drawdown_breaker()
                            if dd_block:
                                self.log.warning("Skip new trade: %s", dd_reason)
                                self.state.set_last_time(last_complete_iso)
                                last_seen = last_complete_iso
                                continue

                            # Regime detection
                            regime = self._check_regime(df)
                            if regime == Regime.CHOPPY:
                                self.log.info("Skip: market regime is CHOPPY — no trading.")
                                self.state.set_last_time(last_complete_iso)
                                last_seen = last_complete_iso
                                continue
                            self.log.info("Market regime: %s", regime.value)

                            # Multi-timeframe check
                            mtf_block, mtf_reason = self._check_mtf_block(last_sig)
                            if mtf_block:
                                self.log.info("Skip: MTF filter — %s", mtf_reason)
                                self.state.set_last_time(last_complete_iso)
                                last_seen = last_complete_iso
                                continue

                            # filters & circuit breakers
                            block, reason = self._check_circuit_breakers(df, filters_cfg, breakers_cfg)
                            if block:
                                self.log.info("Skip new trade: %s", reason)
                                self.state.set_last_time(last_complete_iso)
                                last_seen = last_complete_iso
                                continue

                            side = "long" if last_sig > 0 else "short"
                            entry = last["ask_o"] if side == "long" else last["bid_o"]

                            # --- exits ---
                            use_custom = bool(s.get("use_custom_sl_tp", False))
                            if use_custom and s.get("custom_exit_mode", "") == "prev_bar_extreme":
                                prev = df.iloc[-2]
                                if side == "long":
                                    base_sl = float(prev["bid_l"])
                                    risk_px = entry - base_sl
                                else:
                                    base_sl = float(prev["ask_h"])
                                    risk_px = base_sl - entry

                                min_buf = float(s.get("min_sl_buffer_atr", 0.0)) * float(last["ATR"])
                                if risk_px <= max(min_buf, 1e-8):
                                    if side == "long":
                                        base_sl = entry - max(min_buf, 1e-6)
                                        risk_px = entry - base_sl
                                    else:
                                        base_sl = entry + max(min_buf, 1e-6)
                                        risk_px = base_sl - entry

                                mult = float(s.get("tp_multiple", 2.0))
                                sl = base_sl
                                tp = entry + mult * risk_px if side == "long" else entry - mult * risk_px
                                self.log.info("Custom exits (prev-bar extreme + %.2fR). risk=%.5f", mult, risk_px)
                            else:
                                sl = entry - s["atr_sl_mult"] * last["ATR"] if side == "long" else entry + s["atr_sl_mult"] * last["ATR"]
                                tp = entry + s["atr_tp_mult"] * last["ATR"] if side == "long" else entry - s["atr_tp_mult"] * last["ATR"]

                            # ---- position sizing ----
                            lots = 0.0
                            equity = self._account_equity()
                            if equity and float(r.get("risk_per_trade", 0)) > 0:
                                lots = self._lots_from_risk_pct(self.symbol, float(equity), float(r["risk_per_trade"]), float(entry), float(sl))

                            if lots <= 0:
                                lots = float(r.get("fixed_lots", 0.0))
                            if lots <= 0:
                                si = mt5.symbol_info(self.symbol)
                                lots = float(si.volume_min) if si else 0.01

                            # advisor hook
                            if advisor_enabled:
                                ctx = {
                                    "symbol": self.symbol,
                                    "timeframe": self.tf_str,
                                    "side": side,
                                    "entry": float(entry),
                                    "sl": float(sl),
                                    "tp": float(tp),
                                    "atr": float(last["ATR"]),
                                    "regime": regime.value,
                                }
                                note = advisor_mod.summarize_decision(ctx)
                                if note:
                                    self.log.info("[Advisor] %s", note)

                            res = self._place_market(side, sl, tp, deviation_points=10, lots=lots)
                            self.log.info("Order result: %s", res)

                            # Record to journal
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                self.journal.record_entry(
                                    ticket=res.order,
                                    symbol=self.symbol,
                                    side=side,
                                    entry_price=float(entry),
                                    sl=float(sl),
                                    tp=float(tp),
                                    lots=lots,
                                    strategy=s.get("plugin_module", "ema_cross"),
                                    regime=regime.value,
                                    equity_at_entry=equity,
                                )

                            # Update peak equity
                            if equity > self._peak_equity:
                                self._peak_equity = equity
                        else:
                            self.log.info("Position already open; skipping new trade.")
                    else:
                        self.log.info("No signal or ATR invalid.")

                    self.state.set_last_time(last_complete_iso)
                    last_seen = last_complete_iso

            except Exception as e:
                self.log.exception("Error in live loop: %s", e)
                # Try to reconnect if it looks like a connection issue
                if "initialize" in str(e).lower() or "connection" in str(e).lower():
                    self._try_reconnect()

            # Adaptive polling: sleep until near next bar close
            sleep_secs = self._seconds_to_next_bar()
            time.sleep(sleep_secs)
