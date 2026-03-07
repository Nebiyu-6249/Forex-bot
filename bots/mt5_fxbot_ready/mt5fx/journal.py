# mt5fx/journal.py
"""
SQLite-based trade journal for persistent trade tracking and performance analytics.
Records every trade entry/exit with full context for later analysis.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class TradeJournal:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT,
                side TEXT,
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                sl REAL,
                tp REAL,
                lots REAL,
                pnl_usd REAL,
                result TEXT,
                strategy TEXT,
                regime TEXT,
                equity_at_entry REAL,
                equity_at_exit REAL,
                notes TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                balance REAL,
                open_positions INTEGER
            )
        """)
        self._conn.commit()

    def record_entry(
        self,
        ticket: int,
        symbol: str,
        side: str,
        entry_price: float,
        sl: float,
        tp: float,
        lots: float,
        strategy: str = "",
        regime: str = "",
        equity_at_entry: float = 0.0,
    ):
        self._conn.execute(
            """INSERT INTO trades (ticket, symbol, side, entry_time, entry_price, sl, tp, lots, strategy, regime, equity_at_entry)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticket, symbol, side, datetime.utcnow().isoformat(), entry_price, sl, tp, lots, strategy, regime, equity_at_entry),
        )
        self._conn.commit()

    def record_exit(
        self,
        ticket: int,
        exit_price: float,
        pnl_usd: float,
        result: str,
        equity_at_exit: float = 0.0,
    ):
        self._conn.execute(
            """UPDATE trades SET exit_time=?, exit_price=?, pnl_usd=?, result=?, equity_at_exit=?
               WHERE ticket=? AND exit_time IS NULL""",
            (datetime.utcnow().isoformat(), exit_price, pnl_usd, result, equity_at_exit, ticket),
        )
        self._conn.commit()

    def record_equity_snapshot(self, equity: float, balance: float, open_positions: int):
        self._conn.execute(
            """INSERT INTO equity_snapshots (timestamp, equity, balance, open_positions)
               VALUES (?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), equity, balance, open_positions),
        )
        self._conn.commit()

    def get_recent_trades(self, limit: int = 20) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Compute overall trading statistics from journal."""
        cursor = self._conn.execute(
            "SELECT pnl_usd, result FROM trades WHERE pnl_usd IS NOT NULL"
        )
        rows = cursor.fetchall()
        if not rows:
            return {"total_trades": 0}

        pnls = [r[0] for r in rows]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))

        return {
            "total_trades": len(pnls),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_winner": gross_profit / len(winners) if winners else 0,
            "avg_loser": sum(losers) / len(losers) if losers else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "expectancy": sum(pnls) / len(pnls) if pnls else 0,
        }

    def close(self):
        self._conn.close()
