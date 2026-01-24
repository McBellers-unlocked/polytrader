"""Polymarket CLOB client for order execution."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

import aiohttp

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class OrderResponse:
    """Response from order placement."""

    order_id: str
    success: bool
    status: str
    message: str
    filled_amount: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")


class PolymarketClient:
    """
    Client for Polymarket CLOB API.

    Handles order placement, cancellation, and position tracking.
    Uses the py-clob-client package for signing when in live mode.
    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize the Polymarket client."""
        self.settings = get_settings()
        self._session = session
        self._owns_session = session is None
        self._clob_client: Any = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _get_clob_client(self) -> Any:
        """Get or create the py-clob-client instance."""
        if self._clob_client is None and self.settings.is_live_trading:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                # Initialize with credentials
                self._clob_client = ClobClient(
                    host=self.settings.polymarket_clob_url,
                    chain_id=137,  # Polygon
                    key=self.settings.polymarket_private_key,
                    creds=ApiCreds(
                        api_key=self.settings.polymarket_api_key,
                        api_secret="",
                        api_passphrase="",
                    ) if self.settings.polymarket_api_key else None,
                    funder=self.settings.polymarket_funder,
                )
            except ImportError:
                logger.warning("py-clob-client not installed, using paper trading")
                self._clob_client = None

        return self._clob_client

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        order_type: OrderType = OrderType.LIMIT,
    ) -> OrderResponse:
        """
        Place an order on Polymarket.

        Args:
            token_id: The token to trade
            side: BUY or SELL
            price: Limit price (0-1 for YES tokens)
            size: Order size in shares
            order_type: Order type (only LIMIT supported)

        Returns:
            OrderResponse with result
        """
        if self.settings.is_paper_trading:
            return self._simulate_order(token_id, side, price, size)

        clob = self._get_clob_client()
        if clob is None:
            logger.warning("No CLOB client available, simulating order")
            return self._simulate_order(token_id, side, price, size)

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType

            # Build order args for py-clob-client
            order_args = OrderArgs(
                token_id=token_id,
                price=float(price),
                size=float(size),
                side=side.value,
            )

            # Create and sign the order
            order = clob.create_order(order_args)

            # Submit the order
            response = clob.post_order(order)

            logger.info(
                "Order placed",
                token_id=token_id,
                side=side.value,
                price=str(price),
                size=str(size),
                order_id=response.get("orderID", ""),
            )

            return OrderResponse(
                order_id=response.get("orderID", ""),
                success=response.get("success", False),
                status=response.get("status", "UNKNOWN"),
                message=response.get("message", ""),
            )

        except Exception as e:
            logger.error(
                "Order placement failed",
                token_id=token_id,
                side=side.value,
                error=str(e),
            )
            return OrderResponse(
                order_id="",
                success=False,
                status="ERROR",
                message=str(e),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if self.settings.is_paper_trading:
            logger.info("Paper trading: simulating order cancellation", order_id=order_id)
            return True

        clob = self._get_clob_client()
        if clob is None:
            return True

        try:
            clob.cancel(order_id)
            logger.info("Order cancelled", order_id=order_id)
            return True

        except Exception as e:
            logger.error("Order cancellation failed", order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get status of an order.

        Args:
            order_id: The order ID

        Returns:
            Order status dict
        """
        if self.settings.is_paper_trading:
            return {
                "order_id": order_id,
                "status": "FILLED",
                "filled_amount": "100",
            }

        session = await self._get_session()

        try:
            url = f"{self.settings.polymarket_clob_url}/order/{order_id}"

            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return {"order_id": order_id, "status": "UNKNOWN"}

                return await response.json()

        except aiohttp.ClientError as e:
            logger.warning("Failed to get order status", order_id=order_id, error=str(e))
            return {"order_id": order_id, "status": "UNKNOWN"}

    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current positions.

        Returns:
            List of position dicts
        """
        if self.settings.is_paper_trading:
            return []

        clob = self._get_clob_client()
        if clob is None:
            return []

        try:
            return clob.get_positions()
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            return []

    async def get_balance(self) -> dict[str, Decimal]:
        """
        Get account balances.

        Returns:
            Dict with USDC and other balances
        """
        if self.settings.is_paper_trading:
            return {"USDC": Decimal(str(self.settings.starting_bankroll))}

        clob = self._get_clob_client()
        if clob is None:
            return {"USDC": Decimal("0")}

        try:
            balances = clob.get_balances()
            return {k: Decimal(str(v)) for k, v in balances.items()}
        except Exception as e:
            logger.error("Failed to get balances", error=str(e))
            return {"USDC": Decimal("0")}

    def _simulate_order(
        self,
        token_id: str,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
    ) -> OrderResponse:
        """Simulate an order for paper trading."""
        import uuid

        order_id = f"paper_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Paper trade executed",
            order_id=order_id,
            token_id=token_id,
            side=side.value,
            price=str(price),
            size=str(size),
        )

        return OrderResponse(
            order_id=order_id,
            success=True,
            status="FILLED",
            message="Paper trade simulated",
            filled_amount=size,
            average_price=price,
        )
