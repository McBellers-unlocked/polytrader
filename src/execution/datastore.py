"""Comprehensive SQLite data store for backtesting and analysis.

Stores:
1. All detected opportunities (even if not traded)
2. All trades executed
3. All positions (open and closed)
4. Price snapshots every 5 minutes
5. Forecast snapshots
6. Daily P&L summaries
7. METAR observations

This data is essential for backtesting and strategy refinement.
"""

import json
from dataclasses import asdict
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


class DataStore:
    """
    Comprehensive SQLite data store for all trading data.

    Usage:
        async with DataStore() as store:
            await store.save_opportunity(...)
            await store.save_price_snapshot(...)
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the data store."""
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path
        self._connection: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "DataStore":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Connect to the database and create tables."""
        # Ensure directory exists
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("DataStore connected", path=self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("DataStore closed")

    async def _create_tables(self) -> None:
        """Create all database tables."""
        if not self._connection:
            return

        await self._connection.executescript("""
            -- =====================================================
            -- OPPORTUNITIES: All detected opportunities
            -- =====================================================
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Market info
                condition_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,
                outcome TEXT NOT NULL,

                -- Price data
                market_price REAL NOT NULL,
                fair_probability REAL NOT NULL,
                edge REAL NOT NULL,

                -- Trading decision
                side TEXT NOT NULL,  -- BUY, SELL, HOLD
                suggested_size REAL NOT NULL,
                expected_profit REAL NOT NULL,
                kelly_fraction REAL NOT NULL,

                -- Quality metrics
                model_agreement REAL NOT NULL,
                confidence REAL NOT NULL,

                -- METAR info
                metar_boosted INTEGER NOT NULL DEFAULT 0,
                metar_max_temp REAL,
                metar_confidence REAL,

                -- Was it traded?
                was_traded INTEGER NOT NULL DEFAULT 0,
                trade_blocked_reason TEXT,

                -- Full JSON for analysis
                data JSON
            );

            CREATE INDEX IF NOT EXISTS idx_opp_timestamp ON opportunities(timestamp);
            CREATE INDEX IF NOT EXISTS idx_opp_city ON opportunities(city);
            CREATE INDEX IF NOT EXISTS idx_opp_edge ON opportunities(edge);
            CREATE INDEX IF NOT EXISTS idx_opp_traded ON opportunities(was_traded);

            -- =====================================================
            -- TRADES: All executed trades
            -- =====================================================
            CREATE TABLE IF NOT EXISTS executed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Trade info
                opportunity_id INTEGER,
                condition_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,
                outcome TEXT NOT NULL,

                -- Execution
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                notional REAL NOT NULL,

                -- Edge metrics at time of trade
                fair_probability REAL NOT NULL,
                market_probability REAL NOT NULL,
                edge REAL NOT NULL,
                expected_profit REAL NOT NULL,

                -- Mode
                trading_mode TEXT NOT NULL,  -- paper, semi, auto

                -- Result (filled after resolution)
                actual_outcome TEXT,
                pnl REAL,
                was_correct INTEGER,

                -- Full JSON
                data JSON,

                FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON executed_trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trade_city ON executed_trades(city);
            CREATE INDEX IF NOT EXISTS idx_trade_pnl ON executed_trades(pnl);

            -- =====================================================
            -- POSITIONS: All positions (open and closed)
            -- =====================================================
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Position info
                condition_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,
                outcome TEXT NOT NULL,

                -- Entry
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_size REAL NOT NULL,
                entry_notional REAL NOT NULL,
                entry_edge REAL NOT NULL,

                -- Current/Exit
                current_price REAL NOT NULL,
                current_size REAL NOT NULL,

                -- Status
                status TEXT NOT NULL,  -- open, closed, resolved

                -- Exit (if closed)
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,  -- take_profit, stop_loss, resolution, manual

                -- P&L
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,

                -- Resolution
                resolved_outcome TEXT,
                resolution_time TEXT,

                -- Full JSON
                data JSON
            );

            CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_pos_city ON positions(city);
            CREATE INDEX IF NOT EXISTS idx_pos_target ON positions(target_date);

            -- =====================================================
            -- PRICE_SNAPSHOTS: Market prices every 5 minutes
            -- =====================================================
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Market info
                condition_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,
                outcome TEXT NOT NULL,

                -- Prices
                yes_price REAL NOT NULL,
                no_price REAL NOT NULL,
                best_bid REAL,
                best_ask REAL,
                bid_size REAL,
                ask_size REAL,
                spread REAL,

                -- Market stats
                volume_24h REAL,
                liquidity REAL
            );

            CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_price_token ON price_snapshots(token_id);
            CREATE INDEX IF NOT EXISTS idx_price_city_date ON price_snapshots(city, target_date);

            -- =====================================================
            -- FORECAST_SNAPSHOTS: Ensemble forecasts
            -- =====================================================
            CREATE TABLE IF NOT EXISTS forecast_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Location
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,

                -- Statistics
                n_members INTEGER NOT NULL,
                mean REAL NOT NULL,
                std REAL NOT NULL,
                min REAL NOT NULL,
                max REAL NOT NULL,
                p10 REAL NOT NULL,
                p25 REAL NOT NULL,
                p50 REAL NOT NULL,
                p75 REAL NOT NULL,
                p90 REAL NOT NULL,

                -- Model info
                model_agreement REAL NOT NULL,

                -- Per-model means (JSON)
                model_means JSON,

                -- Full temperatures array (compressed)
                temperatures BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_forecast_timestamp ON forecast_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_forecast_city_date ON forecast_snapshots(city, target_date);

            -- =====================================================
            -- METAR_OBSERVATIONS: METAR data for nowcasting
            -- =====================================================
            CREATE TABLE IF NOT EXISTS metar_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                observation_time TEXT NOT NULL,

                -- Location
                station TEXT NOT NULL,
                city TEXT NOT NULL,

                -- Temperature
                temperature_c REAL NOT NULL,
                temperature_f REAL NOT NULL,

                -- Other data
                dewpoint_c REAL,
                wind_speed_kt INTEGER,
                wind_direction INTEGER,
                visibility_miles REAL,

                -- Raw METAR
                raw_metar TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_metar_timestamp ON metar_observations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metar_city ON metar_observations(city);

            -- =====================================================
            -- DAILY_SUMMARIES: Daily P&L and statistics
            -- =====================================================
            CREATE TABLE IF NOT EXISTS daily_summaries (
                date TEXT PRIMARY KEY,

                -- Bankroll
                starting_bankroll REAL NOT NULL,
                ending_bankroll REAL NOT NULL,

                -- P&L
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL,

                -- Trade stats
                n_opportunities INTEGER NOT NULL,
                n_trades INTEGER NOT NULL,
                n_wins INTEGER NOT NULL,
                n_losses INTEGER NOT NULL,
                win_rate REAL,

                -- Edge stats
                avg_edge REAL,
                total_expected_profit REAL,

                -- METAR stats
                n_metar_constraints INTEGER NOT NULL DEFAULT 0,
                n_metar_trades INTEGER NOT NULL DEFAULT 0,

                -- Risk
                max_drawdown_pct REAL,
                positions_at_close INTEGER,

                -- Full JSON
                data JSON
            );

            -- =====================================================
            -- ARBITRAGE_OPPORTUNITIES: Detected arbitrage
            -- =====================================================
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Market
                condition_id TEXT NOT NULL,
                city TEXT NOT NULL,
                target_date TEXT NOT NULL,

                -- Arbitrage details
                total_yes_price REAL NOT NULL,
                gap REAL NOT NULL,
                profit_margin REAL NOT NULL,
                roi_percent REAL NOT NULL,

                -- Sizing
                required_capital REAL NOT NULL,
                guaranteed_profit REAL NOT NULL,

                -- Execution
                was_executed INTEGER NOT NULL DEFAULT 0,
                execution_time TEXT,
                actual_profit REAL,

                -- Full JSON
                data JSON
            );

            CREATE INDEX IF NOT EXISTS idx_arb_timestamp ON arbitrage_opportunities(timestamp);
            CREATE INDEX IF NOT EXISTS idx_arb_city ON arbitrage_opportunities(city);
        """)
        await self._connection.commit()

    # =========================================================================
    # OPPORTUNITIES
    # =========================================================================

    async def save_opportunity(
        self,
        timestamp: datetime,
        condition_id: str,
        token_id: str,
        city: str,
        target_date: date,
        outcome: str,
        market_price: float,
        fair_probability: float,
        edge: float,
        side: str,
        suggested_size: float,
        expected_profit: float,
        kelly_fraction: float,
        model_agreement: float,
        confidence: float,
        metar_boosted: bool = False,
        metar_max_temp: float | None = None,
        metar_confidence: float | None = None,
        was_traded: bool = False,
        trade_blocked_reason: str | None = None,
        extra_data: dict | None = None,
    ) -> int:
        """Save a detected opportunity."""
        if not self._connection:
            return -1

        cursor = await self._connection.execute("""
            INSERT INTO opportunities (
                timestamp, condition_id, token_id, city, target_date, outcome,
                market_price, fair_probability, edge, side, suggested_size,
                expected_profit, kelly_fraction, model_agreement, confidence,
                metar_boosted, metar_max_temp, metar_confidence,
                was_traded, trade_blocked_reason, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            condition_id,
            token_id,
            city,
            target_date.isoformat(),
            outcome,
            market_price,
            fair_probability,
            edge,
            side,
            suggested_size,
            expected_profit,
            kelly_fraction,
            model_agreement,
            confidence,
            1 if metar_boosted else 0,
            metar_max_temp,
            metar_confidence,
            1 if was_traded else 0,
            trade_blocked_reason,
            json.dumps(extra_data or {}),
        ))
        await self._connection.commit()
        return cursor.lastrowid or -1

    async def get_opportunities(
        self,
        city: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        min_edge: float | None = None,
        traded_only: bool = False,
    ) -> list[dict]:
        """Query opportunities with filters."""
        if not self._connection:
            return []

        query = "SELECT * FROM opportunities WHERE 1=1"
        params: list[Any] = []

        if city:
            query += " AND city = ?"
            params.append(city)
        if start_date:
            query += " AND date(target_date) >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date(target_date) <= ?"
            params.append(end_date.isoformat())
        if min_edge:
            query += " AND abs(edge) >= ?"
            params.append(min_edge)
        if traded_only:
            query += " AND was_traded = 1"

        query += " ORDER BY timestamp DESC"

        results = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # TRADES
    # =========================================================================

    async def save_trade(
        self,
        timestamp: datetime,
        opportunity_id: int | None,
        condition_id: str,
        token_id: str,
        city: str,
        target_date: date,
        outcome: str,
        side: str,
        price: float,
        size: float,
        fair_probability: float,
        market_probability: float,
        edge: float,
        expected_profit: float,
        trading_mode: str,
        extra_data: dict | None = None,
    ) -> int:
        """Save an executed trade."""
        if not self._connection:
            return -1

        notional = price * size

        cursor = await self._connection.execute("""
            INSERT INTO executed_trades (
                timestamp, opportunity_id, condition_id, token_id, city,
                target_date, outcome, side, price, size, notional,
                fair_probability, market_probability, edge, expected_profit,
                trading_mode, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            opportunity_id,
            condition_id,
            token_id,
            city,
            target_date.isoformat(),
            outcome,
            side,
            price,
            size,
            notional,
            fair_probability,
            market_probability,
            edge,
            expected_profit,
            trading_mode,
            json.dumps(extra_data or {}),
        ))
        await self._connection.commit()
        return cursor.lastrowid or -1

    async def update_trade_result(
        self,
        trade_id: int,
        actual_outcome: str,
        pnl: float,
        was_correct: bool,
    ) -> None:
        """Update trade with resolution result."""
        if not self._connection:
            return

        await self._connection.execute("""
            UPDATE executed_trades
            SET actual_outcome = ?, pnl = ?, was_correct = ?
            WHERE id = ?
        """, (actual_outcome, pnl, 1 if was_correct else 0, trade_id))
        await self._connection.commit()

    async def get_trades(
        self,
        city: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        trading_mode: str | None = None,
    ) -> list[dict]:
        """Query trades with filters."""
        if not self._connection:
            return []

        query = "SELECT * FROM executed_trades WHERE 1=1"
        params: list[Any] = []

        if city:
            query += " AND city = ?"
            params.append(city)
        if start_date:
            query += " AND date(target_date) >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date(target_date) <= ?"
            params.append(end_date.isoformat())
        if trading_mode:
            query += " AND trading_mode = ?"
            params.append(trading_mode)

        query += " ORDER BY timestamp DESC"

        results = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # POSITIONS
    # =========================================================================

    async def save_position(
        self,
        condition_id: str,
        token_id: str,
        city: str,
        target_date: date,
        outcome: str,
        entry_time: datetime,
        entry_price: float,
        entry_size: float,
        entry_edge: float,
        current_price: float,
        status: str = "open",
        extra_data: dict | None = None,
    ) -> int:
        """Save a new position."""
        if not self._connection:
            return -1

        entry_notional = entry_price * entry_size

        cursor = await self._connection.execute("""
            INSERT INTO positions (
                condition_id, token_id, city, target_date, outcome,
                entry_time, entry_price, entry_size, entry_notional, entry_edge,
                current_price, current_size, status, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            condition_id,
            token_id,
            city,
            target_date.isoformat(),
            outcome,
            entry_time.isoformat(),
            entry_price,
            entry_size,
            entry_notional,
            entry_edge,
            current_price,
            entry_size,
            status,
            json.dumps(extra_data or {}),
        ))
        await self._connection.commit()
        return cursor.lastrowid or -1

    async def close_position(
        self,
        position_id: int,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        realized_pnl: float,
    ) -> None:
        """Close a position."""
        if not self._connection:
            return

        await self._connection.execute("""
            UPDATE positions
            SET status = 'closed', exit_time = ?, exit_price = ?,
                exit_reason = ?, realized_pnl = ?, current_size = 0
            WHERE id = ?
        """, (exit_time.isoformat(), exit_price, exit_reason, realized_pnl, position_id))
        await self._connection.commit()

    async def resolve_position(
        self,
        position_id: int,
        resolved_outcome: str,
        resolution_time: datetime,
        realized_pnl: float,
    ) -> None:
        """Mark position as resolved."""
        if not self._connection:
            return

        await self._connection.execute("""
            UPDATE positions
            SET status = 'resolved', resolved_outcome = ?,
                resolution_time = ?, realized_pnl = ?, current_size = 0
            WHERE id = ?
        """, (resolved_outcome, resolution_time.isoformat(), realized_pnl, position_id))
        await self._connection.commit()

    async def get_open_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._connection:
            return []

        results = []
        async with self._connection.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time"
        ) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # PRICE SNAPSHOTS
    # =========================================================================

    async def save_price_snapshot(
        self,
        timestamp: datetime,
        condition_id: str,
        token_id: str,
        city: str,
        target_date: date,
        outcome: str,
        yes_price: float,
        no_price: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
        bid_size: float | None = None,
        ask_size: float | None = None,
        spread: float | None = None,
        volume_24h: float | None = None,
        liquidity: float | None = None,
    ) -> None:
        """Save a price snapshot."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT INTO price_snapshots (
                timestamp, condition_id, token_id, city, target_date, outcome,
                yes_price, no_price, best_bid, best_ask, bid_size, ask_size,
                spread, volume_24h, liquidity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            condition_id,
            token_id,
            city,
            target_date.isoformat(),
            outcome,
            yes_price,
            no_price,
            best_bid,
            best_ask,
            bid_size,
            ask_size,
            spread,
            volume_24h,
            liquidity,
        ))
        await self._connection.commit()

    async def save_market_prices(
        self,
        timestamp: datetime,
        market_data: list[dict],
    ) -> None:
        """Save price snapshots for multiple buckets."""
        if not self._connection:
            return

        for data in market_data:
            await self.save_price_snapshot(timestamp=timestamp, **data)

    async def get_price_history(
        self,
        token_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict]:
        """Get price history for a token."""
        if not self._connection:
            return []

        query = "SELECT * FROM price_snapshots WHERE token_id = ?"
        params: list[Any] = [token_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp"

        results = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # FORECAST SNAPSHOTS
    # =========================================================================

    async def save_forecast_snapshot(
        self,
        timestamp: datetime,
        city: str,
        target_date: date,
        n_members: int,
        mean: float,
        std: float,
        min_temp: float,
        max_temp: float,
        p10: float,
        p25: float,
        p50: float,
        p75: float,
        p90: float,
        model_agreement: float,
        model_means: dict | None = None,
        temperatures: bytes | None = None,
    ) -> None:
        """Save a forecast snapshot."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT INTO forecast_snapshots (
                timestamp, city, target_date, n_members, mean, std, min, max,
                p10, p25, p50, p75, p90, model_agreement, model_means, temperatures
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            city,
            target_date.isoformat(),
            n_members,
            mean,
            std,
            min_temp,
            max_temp,
            p10,
            p25,
            p50,
            p75,
            p90,
            model_agreement,
            json.dumps(model_means) if model_means else None,
            temperatures,
        ))
        await self._connection.commit()

    async def get_forecast_history(
        self,
        city: str,
        target_date: date,
    ) -> list[dict]:
        """Get forecast history for a city/date."""
        if not self._connection:
            return []

        results = []
        async with self._connection.execute("""
            SELECT * FROM forecast_snapshots
            WHERE city = ? AND target_date = ?
            ORDER BY timestamp
        """, (city, target_date.isoformat())) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # METAR OBSERVATIONS
    # =========================================================================

    async def save_metar_observation(
        self,
        timestamp: datetime,
        observation_time: datetime,
        station: str,
        city: str,
        temperature_c: float,
        temperature_f: float,
        dewpoint_c: float | None = None,
        wind_speed_kt: int | None = None,
        wind_direction: int | None = None,
        visibility_miles: float | None = None,
        raw_metar: str | None = None,
    ) -> None:
        """Save a METAR observation."""
        if not self._connection:
            return

        await self._connection.execute("""
            INSERT INTO metar_observations (
                timestamp, observation_time, station, city,
                temperature_c, temperature_f, dewpoint_c,
                wind_speed_kt, wind_direction, visibility_miles, raw_metar
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            observation_time.isoformat(),
            station,
            city,
            temperature_c,
            temperature_f,
            dewpoint_c,
            wind_speed_kt,
            wind_direction,
            visibility_miles,
            raw_metar,
        ))
        await self._connection.commit()

    async def get_daily_max_temp(
        self,
        city: str,
        target_date: date,
    ) -> float | None:
        """Get max observed temperature for a city/date."""
        if not self._connection:
            return None

        async with self._connection.execute("""
            SELECT MAX(temperature_f) FROM metar_observations
            WHERE city = ? AND date(observation_time) = ?
        """, (city, target_date.isoformat())) as cursor:
            row = await cursor.fetchone()
            if row and row[0] is not None:
                return float(row[0])
        return None

    # =========================================================================
    # DAILY SUMMARIES
    # =========================================================================

    async def save_daily_summary(
        self,
        summary_date: date,
        starting_bankroll: float,
        ending_bankroll: float,
        realized_pnl: float,
        unrealized_pnl: float,
        n_opportunities: int,
        n_trades: int,
        n_wins: int,
        n_losses: int,
        avg_edge: float | None = None,
        total_expected_profit: float | None = None,
        n_metar_constraints: int = 0,
        n_metar_trades: int = 0,
        max_drawdown_pct: float | None = None,
        positions_at_close: int = 0,
        extra_data: dict | None = None,
    ) -> None:
        """Save daily summary."""
        if not self._connection:
            return

        total_pnl = realized_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / starting_bankroll * 100) if starting_bankroll > 0 else 0
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else None

        await self._connection.execute("""
            INSERT OR REPLACE INTO daily_summaries (
                date, starting_bankroll, ending_bankroll, realized_pnl,
                unrealized_pnl, total_pnl, total_pnl_pct, n_opportunities,
                n_trades, n_wins, n_losses, win_rate, avg_edge,
                total_expected_profit, n_metar_constraints, n_metar_trades,
                max_drawdown_pct, positions_at_close, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_date.isoformat(),
            starting_bankroll,
            ending_bankroll,
            realized_pnl,
            unrealized_pnl,
            total_pnl,
            total_pnl_pct,
            n_opportunities,
            n_trades,
            n_wins,
            n_losses,
            win_rate,
            avg_edge,
            total_expected_profit,
            n_metar_constraints,
            n_metar_trades,
            max_drawdown_pct,
            positions_at_close,
            json.dumps(extra_data or {}),
        ))
        await self._connection.commit()

    async def get_daily_summary(self, summary_date: date) -> dict | None:
        """Get daily summary."""
        if not self._connection:
            return None

        async with self._connection.execute(
            "SELECT * FROM daily_summaries WHERE date = ?",
            (summary_date.isoformat(),)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None

    async def get_daily_summaries(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get daily summaries for a date range."""
        if not self._connection:
            return []

        query = "SELECT * FROM daily_summaries WHERE 1=1"
        params: list[Any] = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date"

        results = []
        async with self._connection.execute(query, params) as cursor:
            async for row in cursor:
                results.append(dict(row))
        return results

    # =========================================================================
    # ARBITRAGE
    # =========================================================================

    async def save_arbitrage_opportunity(
        self,
        timestamp: datetime,
        condition_id: str,
        city: str,
        target_date: date,
        total_yes_price: float,
        gap: float,
        profit_margin: float,
        roi_percent: float,
        required_capital: float,
        guaranteed_profit: float,
        was_executed: bool = False,
        extra_data: dict | None = None,
    ) -> int:
        """Save an arbitrage opportunity."""
        if not self._connection:
            return -1

        cursor = await self._connection.execute("""
            INSERT INTO arbitrage_opportunities (
                timestamp, condition_id, city, target_date, total_yes_price,
                gap, profit_margin, roi_percent, required_capital,
                guaranteed_profit, was_executed, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            condition_id,
            city,
            target_date.isoformat(),
            total_yes_price,
            gap,
            profit_margin,
            roi_percent,
            required_capital,
            guaranteed_profit,
            1 if was_executed else 0,
            json.dumps(extra_data or {}),
        ))
        await self._connection.commit()
        return cursor.lastrowid or -1

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def get_statistics(self) -> dict[str, Any]:
        """Get overall statistics."""
        if not self._connection:
            return {}

        stats: dict[str, Any] = {}

        # Total opportunities
        async with self._connection.execute(
            "SELECT COUNT(*) FROM opportunities"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_opportunities"] = row[0] if row else 0

        # Total trades
        async with self._connection.execute(
            "SELECT COUNT(*) FROM executed_trades"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_trades"] = row[0] if row else 0

        # Win rate
        async with self._connection.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM executed_trades
            WHERE was_correct IS NOT NULL
        """) as cursor:
            row = await cursor.fetchone()
            if row and row[0] > 0:
                stats["resolved_trades"] = row[0]
                stats["wins"] = row[1]
                stats["win_rate"] = row[1] / row[0] * 100

        # Total P&L
        async with self._connection.execute(
            "SELECT SUM(pnl) FROM executed_trades WHERE pnl IS NOT NULL"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_pnl"] = row[0] if row and row[0] else 0

        # METAR stats
        async with self._connection.execute(
            "SELECT COUNT(*) FROM opportunities WHERE metar_boosted = 1"
        ) as cursor:
            row = await cursor.fetchone()
            stats["metar_opportunities"] = row[0] if row else 0

        return stats


async def demo_datastore():
    """Demonstrate the data store."""
    print(f"\n{'='*60}")
    print("DataStore Demo")
    print(f"{'='*60}\n")

    async with DataStore(db_path=":memory:") as store:
        # Save an opportunity
        opp_id = await store.save_opportunity(
            timestamp=datetime.utcnow(),
            condition_id="test_condition",
            token_id="test_token",
            city="New York City",
            target_date=date.today(),
            outcome="52-56°F",
            market_price=0.25,
            fair_probability=0.35,
            edge=0.40,
            side="BUY",
            suggested_size=100.0,
            expected_profit=10.0,
            kelly_fraction=0.05,
            model_agreement=0.75,
            confidence=0.70,
            metar_boosted=True,
            metar_max_temp=52.0,
            metar_confidence=0.80,
        )
        print(f"Saved opportunity ID: {opp_id}")

        # Save a trade
        trade_id = await store.save_trade(
            timestamp=datetime.utcnow(),
            opportunity_id=opp_id,
            condition_id="test_condition",
            token_id="test_token",
            city="New York City",
            target_date=date.today(),
            outcome="52-56°F",
            side="BUY",
            price=0.25,
            size=100.0,
            fair_probability=0.35,
            market_probability=0.25,
            edge=0.40,
            expected_profit=10.0,
            trading_mode="paper",
        )
        print(f"Saved trade ID: {trade_id}")

        # Save a forecast snapshot
        await store.save_forecast_snapshot(
            timestamp=datetime.utcnow(),
            city="New York City",
            target_date=date.today(),
            n_members=122,
            mean=55.0,
            std=4.5,
            min_temp=45.0,
            max_temp=65.0,
            p10=49.0,
            p25=52.0,
            p50=55.0,
            p75=58.0,
            p90=61.0,
            model_agreement=0.75,
            model_means={"icon": 54.5, "gfs": 55.5, "ecmwf": 55.0},
        )
        print("Saved forecast snapshot")

        # Save daily summary
        await store.save_daily_summary(
            summary_date=date.today(),
            starting_bankroll=1500.0,
            ending_bankroll=1550.0,
            realized_pnl=50.0,
            unrealized_pnl=0.0,
            n_opportunities=15,
            n_trades=3,
            n_wins=2,
            n_losses=1,
            avg_edge=0.25,
            n_metar_constraints=1,
            n_metar_trades=2,
        )
        print("Saved daily summary")

        # Get statistics
        stats = await store.get_statistics()
        print(f"\nStatistics: {stats}")

        # Query opportunities
        opps = await store.get_opportunities(city="New York City")
        print(f"\nOpportunities: {len(opps)}")

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_datastore())
