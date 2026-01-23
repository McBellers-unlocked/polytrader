"""SQLite persistence for orders and positions."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite

from src.config import get_settings
from src.execution.orders import Order, OrderStatus, OrderSide
from src.execution.positions import Position
from src.logging import get_logger

logger = get_logger(__name__)


class Database:
    """
    SQLite database for persisting orders, positions, and trades.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the database."""
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Connect to the database and create tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info("Database connected", path=self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if not self._connection:
            return

        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                size TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                submitted_at TEXT,
                filled_at TEXT,
                cancelled_at TEXT,
                filled_size TEXT NOT NULL,
                average_fill_price TEXT NOT NULL,
                exchange_order_id TEXT,
                error_message TEXT,
                signal_edge REAL,
                signal_confidence REAL,
                bucket_outcome TEXT,
                data JSON
            );

            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                token_id TEXT UNIQUE NOT NULL,
                condition_id TEXT NOT NULL,
                bucket_outcome TEXT,
                city TEXT,
                target_date TEXT,
                size TEXT NOT NULL,
                avg_entry_price TEXT NOT NULL,
                current_price TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                data JSON
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                size TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pnl TEXT,
                data JSON
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                starting_bankroll TEXT NOT NULL,
                ending_bankroll TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                trades INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                data JSON
            );

            CREATE INDEX IF NOT EXISTS idx_orders_token ON orders(token_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token_id);
            CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_id);
        """)
        await self._connection.commit()

    async def save_order(self, order: Order) -> None:
        """Save or update an order."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT OR REPLACE INTO orders (
                id, token_id, condition_id, side, price, size, status,
                created_at, submitted_at, filled_at, cancelled_at,
                filled_size, average_fill_price, exchange_order_id,
                error_message, signal_edge, signal_confidence,
                bucket_outcome, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.id,
            order.token_id,
            order.condition_id,
            order.side.value,
            str(order.price),
            str(order.size),
            order.status.value,
            order.created_at.isoformat(),
            order.submitted_at.isoformat() if order.submitted_at else None,
            order.filled_at.isoformat() if order.filled_at else None,
            order.cancelled_at.isoformat() if order.cancelled_at else None,
            str(order.filled_size),
            str(order.average_fill_price),
            order.exchange_order_id,
            order.error_message,
            order.signal_edge,
            order.signal_confidence,
            order.bucket_outcome,
            json.dumps(order.to_dict()),
        ))
        await self._connection.commit()

    async def get_order(self, order_id: str) -> Order | None:
        """Get an order by ID."""
        if not self._connection:
            return None

        async with self._connection.execute(
            "SELECT data FROM orders WHERE id = ?", (order_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Order.from_dict(json.loads(row[0]))
        return None

    async def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        if not self._connection:
            return []

        orders = []
        async with self._connection.execute(
            "SELECT data FROM orders WHERE status IN (?, ?, ?)",
            (OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value)
        ) as cursor:
            async for row in cursor:
                orders.append(Order.from_dict(json.loads(row[0])))
        return orders

    async def get_orders_by_token(self, token_id: str) -> list[Order]:
        """Get orders for a token."""
        if not self._connection:
            return []

        orders = []
        async with self._connection.execute(
            "SELECT data FROM orders WHERE token_id = ? ORDER BY created_at DESC",
            (token_id,)
        ) as cursor:
            async for row in cursor:
                orders.append(Order.from_dict(json.loads(row[0])))
        return orders

    async def save_position(self, position: Position) -> None:
        """Save or update a position."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT OR REPLACE INTO positions (
                id, token_id, condition_id, bucket_outcome, city, target_date,
                size, avg_entry_price, current_price, opened_at, last_updated,
                realized_pnl, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.id,
            position.token_id,
            position.condition_id,
            position.bucket_outcome,
            position.city,
            position.target_date,
            str(position.size),
            str(position.avg_entry_price),
            str(position.current_price),
            position.opened_at.isoformat(),
            position.last_updated.isoformat(),
            str(position.realized_pnl),
            json.dumps(position.to_dict()),
        ))
        await self._connection.commit()

    async def get_position(self, token_id: str) -> Position | None:
        """Get a position by token ID."""
        if not self._connection:
            return None

        async with self._connection.execute(
            "SELECT data FROM positions WHERE token_id = ?", (token_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Position.from_dict(json.loads(row[0]))
        return None

    async def get_all_positions(self) -> list[Position]:
        """Get all positions."""
        if not self._connection:
            return []

        positions = []
        async with self._connection.execute(
            "SELECT data FROM positions WHERE CAST(size AS REAL) > 0"
        ) as cursor:
            async for row in cursor:
                positions.append(Position.from_dict(json.loads(row[0])))
        return positions

    async def record_trade(
        self,
        order: Order,
        size: Decimal,
        price: Decimal,
        pnl: Decimal | None = None,
    ) -> None:
        """Record a trade execution."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT INTO trades (
                order_id, token_id, condition_id, side, price, size,
                timestamp, pnl, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.id,
            order.token_id,
            order.condition_id,
            order.side.value,
            str(price),
            str(size),
            datetime.utcnow().isoformat(),
            str(pnl) if pnl else None,
            json.dumps({
                "bucket_outcome": order.bucket_outcome,
                "signal_edge": order.signal_edge,
            }),
        ))
        await self._connection.commit()

    async def save_daily_stats(
        self,
        date: str,
        starting_bankroll: Decimal,
        ending_bankroll: Decimal,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        trades: int,
        wins: int,
        losses: int,
    ) -> None:
        """Save daily statistics."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT OR REPLACE INTO daily_stats (
                date, starting_bankroll, ending_bankroll, realized_pnl,
                unrealized_pnl, trades, wins, losses, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            str(starting_bankroll),
            str(ending_bankroll),
            str(realized_pnl),
            str(unrealized_pnl),
            trades,
            wins,
            losses,
            json.dumps({}),
        ))
        await self._connection.commit()

    async def get_daily_stats(self, date: str) -> dict[str, Any] | None:
        """Get daily statistics."""
        if not self._connection:
            return None

        async with self._connection.execute(
            "SELECT * FROM daily_stats WHERE date = ?", (date,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "date": row[0],
                    "starting_bankroll": Decimal(row[1]),
                    "ending_bankroll": Decimal(row[2]),
                    "realized_pnl": Decimal(row[3]),
                    "unrealized_pnl": Decimal(row[4]),
                    "trades": row[5],
                    "wins": row[6],
                    "losses": row[7],
                }
        return None
