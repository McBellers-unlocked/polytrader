"""Execution engine for order management."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.config import get_settings, TradingMode
from src.markets.client import PolymarketClient, OrderSide as ClientOrderSide
from src.execution.orders import Order, OrderStatus, OrderSide
from src.execution.positions import Position, PositionTracker
from src.execution.persistence import Database
from src.strategy.edge import TradeSignal, SignalType
from src.risk.manager import RiskManager
from src.logging import get_logger

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Manages order execution and position tracking.

    Handles:
    - Order creation and submission
    - Order timeout and cancellation
    - Position updates
    - P&L tracking
    """

    def __init__(
        self,
        client: PolymarketClient | None = None,
        risk_manager: RiskManager | None = None,
    ):
        """Initialize the execution engine."""
        self.settings = get_settings()
        self.client = client or PolymarketClient()
        self.risk_manager = risk_manager or RiskManager()
        self.positions = PositionTracker()
        self.db = Database()
        self._active_orders: dict[str, Order] = {}
        self._order_tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        """Start the execution engine."""
        await self.db.connect()

        # Load existing positions
        positions = await self.db.get_all_positions()
        self.positions.load_positions(positions)

        # Load active orders
        orders = await self.db.get_active_orders()
        for order in orders:
            self._active_orders[order.id] = order

        logger.info(
            "Execution engine started",
            n_positions=len(self.positions.get_open_positions()),
            n_active_orders=len(self._active_orders),
        )

    async def stop(self) -> None:
        """Stop the execution engine."""
        # Cancel all pending orders
        for order_id, task in self._order_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel active orders on exchange
        for order in self._active_orders.values():
            if order.is_active:
                await self.cancel_order(order.id)

        await self.client.close()
        await self.db.close()
        logger.info("Execution engine stopped")

    async def execute_signal(self, signal: TradeSignal) -> Order | None:
        """
        Execute a trade signal.

        Args:
            signal: The trade signal to execute

        Returns:
            Order if created, None otherwise
        """
        # Run risk checks
        checks = self.risk_manager.check_trade(signal)

        if not self.risk_manager.is_trade_allowed(checks):
            logger.warning(
                "Trade blocked by risk manager",
                signal=signal.bucket.outcome,
                checks=[c.to_dict() for c in checks if c.is_blocked],
            )
            return None

        # Adjust size based on risk (in dollars)
        adjusted_size_dollars = self.risk_manager.adjust_position_size(
            signal.suggested_size, checks
        )

        if adjusted_size_dollars <= 0:
            logger.info("Trade size reduced to zero", signal=signal.bucket.outcome)
            return None

        # Determine order side
        if signal.signal_type == SignalType.BUY_YES:
            side = OrderSide.BUY
            price = Decimal(str(signal.market_price))
        else:
            side = OrderSide.SELL
            price = Decimal(str(1 - signal.market_price))

        # Convert dollars to shares (Polymarket orders are in shares, not dollars)
        # shares = dollars / price_per_share
        if price <= 0:
            logger.warning("Invalid price for order", price=str(price))
            return None

        shares = adjusted_size_dollars / price

        # Polymarket requires minimum $1 order value
        # If our order is too small, skip it
        order_value = shares * price
        if order_value < Decimal("1"):
            logger.info(
                "Order value below $1 minimum, skipping",
                outcome=signal.bucket.outcome,
                order_value=str(order_value),
            )
            return None

        # Create order (size is in shares)
        order = Order(
            token_id=signal.bucket.token_id,
            condition_id=signal.market.condition_id,
            side=side,
            price=price,
            size=shares,
            signal_edge=signal.edge,
            signal_confidence=signal.confidence,
            bucket_outcome=signal.bucket.outcome,
        )

        # Request confirmation if in semi mode
        if self.settings.trading_mode == TradingMode.SEMI:
            if not await self._request_confirmation(order, signal):
                logger.info("Trade rejected by user", order_id=order.id)
                return None

        # Submit order
        return await self.submit_order(order)

    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange.

        Args:
            order: The order to submit

        Returns:
            Updated order with exchange response
        """
        logger.info(
            "Submitting order",
            order_id=order.id,
            token_id=order.token_id,
            side=order.side.value,
            price=str(order.price),
            size=str(order.size),
        )

        # Map order side
        client_side = (
            ClientOrderSide.BUY if order.side == OrderSide.BUY
            else ClientOrderSide.SELL
        )

        # Submit to exchange
        response = await self.client.place_order(
            token_id=order.token_id,
            side=client_side,
            price=order.price,
            size=order.size,
        )

        if response.success:
            order.mark_submitted(response.order_id)

            # For paper trading, mark as filled immediately
            if self.settings.is_paper_trading:
                order.mark_filled(order.size, order.price)
                await self._on_order_filled(order)
            else:
                # Start timeout monitoring
                self._active_orders[order.id] = order
                task = asyncio.create_task(self._monitor_order(order))
                self._order_tasks[order.id] = task
        else:
            order.mark_rejected(response.message)

        await self.db.save_order(order)

        logger.info(
            "Order submitted",
            order_id=order.id,
            status=order.status.value,
            exchange_order_id=order.exchange_order_id,
        )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancelled successfully
        """
        order = self._active_orders.get(order_id)
        if not order:
            order = await self.db.get_order(order_id)

        if not order or not order.is_active:
            return False

        # Cancel on exchange
        success = await self.client.cancel_order(order.exchange_order_id)

        if success:
            order.mark_cancelled()

            # Cancel monitoring task
            if order_id in self._order_tasks:
                self._order_tasks[order_id].cancel()
                del self._order_tasks[order_id]

            if order_id in self._active_orders:
                del self._active_orders[order_id]

        await self.db.save_order(order)

        logger.info(
            "Order cancelled",
            order_id=order_id,
            success=success,
        )

        return success

    async def _monitor_order(self, order: Order) -> None:
        """Monitor an order for fill or timeout."""
        timeout = self.settings.order_timeout_seconds

        try:
            while order.is_active:
                # Check for timeout
                if order.age_seconds > timeout:
                    logger.info(
                        "Order timeout, cancelling",
                        order_id=order.id,
                        age_seconds=order.age_seconds,
                    )
                    order.mark_expired()
                    await self.client.cancel_order(order.exchange_order_id)
                    break

                # Check order status
                status = await self.client.get_order_status(order.exchange_order_id)

                if status.get("status") == "FILLED":
                    filled = Decimal(str(status.get("filled_amount", order.size)))
                    price = Decimal(str(status.get("average_price", order.price)))
                    order.mark_filled(filled, price)
                    await self._on_order_filled(order)
                    break
                elif status.get("status") == "CANCELLED":
                    order.mark_cancelled()
                    break

                await asyncio.sleep(5)  # Check every 5 seconds

        except asyncio.CancelledError:
            pass
        finally:
            if order.id in self._active_orders:
                del self._active_orders[order.id]
            await self.db.save_order(order)

    async def _on_order_filled(self, order: Order) -> None:
        """Handle order fill."""
        # Update position
        position = self.positions.get_or_create_position(
            token_id=order.token_id,
            condition_id=order.condition_id,
            bucket_outcome=order.bucket_outcome,
        )

        if order.side == OrderSide.BUY:
            position.add_to_position(order.filled_size, order.average_fill_price)
        else:
            pnl = position.reduce_position(order.filled_size, order.average_fill_price)
            self.risk_manager.record_trade(order.token_id, -order.filled_size, pnl)

        # Update position price
        position.update_price(order.average_fill_price)

        # Save to database
        await self.db.save_position(position)
        await self.db.record_trade(
            order,
            order.filled_size,
            order.average_fill_price,
            position.realized_pnl if order.side == OrderSide.SELL else None,
        )

        # Update risk manager
        self.risk_manager.record_trade(
            order.token_id,
            order.filled_size if order.side == OrderSide.BUY else -order.filled_size,
        )

        logger.info(
            "Order filled",
            order_id=order.id,
            filled_size=str(order.filled_size),
            average_price=str(order.average_fill_price),
            position_size=str(position.size),
        )

    async def _request_confirmation(
        self,
        order: Order,
        signal: TradeSignal,
    ) -> bool:
        """Request user confirmation for a trade (semi-auto mode)."""
        print("\n" + "=" * 60)
        print("TRADE CONFIRMATION REQUIRED")
        print("=" * 60)
        print(f"  Market: {signal.market.question[:50]}...")
        print(f"  Outcome: {signal.bucket.outcome}")
        print(f"  Side: {order.side.value}")
        print(f"  Price: ${order.price}")
        print(f"  Size: {order.size} shares")
        print(f"  Edge: {signal.edge:.1%}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print("=" * 60)

        try:
            response = input("Execute trade? [y/N]: ").strip().lower()
            return response == "y"
        except (EOFError, KeyboardInterrupt):
            return False

    async def update_positions(self) -> None:
        """Update all position prices from market."""
        for position in self.positions.get_open_positions():
            # Fetch current price from orderbook
            orderbook = await self.client.get_order_status(position.token_id)
            if "price" in orderbook:
                position.update_price(Decimal(str(orderbook["price"])))
                await self.db.save_position(position)

    def get_status(self) -> dict[str, Any]:
        """Get execution engine status."""
        return {
            "active_orders": len(self._active_orders),
            "open_positions": len(self.positions.get_open_positions()),
            "total_position_value": str(self.positions.total_value),
            "total_unrealized_pnl": str(self.positions.total_unrealized_pnl),
            "total_realized_pnl": str(self.positions.total_realized_pnl),
            "risk_status": self.risk_manager.get_status(),
        }
