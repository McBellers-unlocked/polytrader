"""Order data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
import uuid


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "PENDING"  # Order created but not submitted
    SUBMITTED = "SUBMITTED"  # Submitted to exchange
    PARTIAL = "PARTIAL"  # Partially filled
    FILLED = "FILLED"  # Completely filled
    CANCELLED = "CANCELLED"  # Cancelled
    REJECTED = "REJECTED"  # Rejected by exchange
    EXPIRED = "EXPIRED"  # Expired (timeout)


@dataclass
class Order:
    """
    A trading order.

    Tracks the full lifecycle from creation to fill/cancel.
    """

    id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:12]}")
    token_id: str = ""
    condition_id: str = ""
    side: OrderSide = OrderSide.BUY
    price: Decimal = Decimal("0")
    size: Decimal = Decimal("0")
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    filled_size: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    exchange_order_id: str = ""
    error_message: str = ""

    # Metadata
    signal_edge: float = 0.0
    signal_confidence: float = 0.0
    bucket_outcome: str = ""

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
        )

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled, cancelled, etc.)."""
        return not self.is_active

    @property
    def unfilled_size(self) -> Decimal:
        """Get unfilled size."""
        return self.size - self.filled_size

    @property
    def fill_rate(self) -> float:
        """Get fill rate percentage."""
        if self.size > 0:
            return float(self.filled_size / self.size)
        return 0.0

    @property
    def notional_value(self) -> Decimal:
        """Get notional value of order."""
        return self.price * self.size

    @property
    def filled_value(self) -> Decimal:
        """Get value of filled portion."""
        return self.average_fill_price * self.filled_size

    @property
    def age_seconds(self) -> float:
        """Get order age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    def mark_submitted(self, exchange_order_id: str) -> None:
        """Mark order as submitted."""
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.utcnow()
        self.exchange_order_id = exchange_order_id

    def mark_partial_fill(self, filled: Decimal, price: Decimal) -> None:
        """Record a partial fill."""
        self.status = OrderStatus.PARTIAL
        # Update weighted average fill price
        total_value = self.average_fill_price * self.filled_size + price * filled
        self.filled_size += filled
        if self.filled_size > 0:
            self.average_fill_price = total_value / self.filled_size

    def mark_filled(self, filled: Decimal | None = None, price: Decimal | None = None) -> None:
        """Mark order as completely filled."""
        if filled is not None and price is not None:
            self.mark_partial_fill(filled, price)
        self.status = OrderStatus.FILLED
        self.filled_at = datetime.utcnow()

    def mark_cancelled(self) -> None:
        """Mark order as cancelled."""
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()

    def mark_rejected(self, message: str) -> None:
        """Mark order as rejected."""
        self.status = OrderStatus.REJECTED
        self.error_message = message

    def mark_expired(self) -> None:
        """Mark order as expired."""
        self.status = OrderStatus.EXPIRED
        self.cancelled_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "side": self.side.value,
            "price": str(self.price),
            "size": str(self.size),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_size": str(self.filled_size),
            "average_fill_price": str(self.average_fill_price),
            "exchange_order_id": self.exchange_order_id,
            "error_message": self.error_message,
            "signal_edge": self.signal_edge,
            "bucket_outcome": self.bucket_outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Order":
        """Create from dictionary."""
        order = cls(
            id=data.get("id", ""),
            token_id=data.get("token_id", ""),
            condition_id=data.get("condition_id", ""),
            side=OrderSide(data.get("side", "BUY")),
            price=Decimal(data.get("price", "0")),
            size=Decimal(data.get("size", "0")),
            status=OrderStatus(data.get("status", "PENDING")),
            filled_size=Decimal(data.get("filled_size", "0")),
            average_fill_price=Decimal(data.get("average_fill_price", "0")),
            exchange_order_id=data.get("exchange_order_id", ""),
            error_message=data.get("error_message", ""),
            signal_edge=data.get("signal_edge", 0.0),
            bucket_outcome=data.get("bucket_outcome", ""),
        )

        if data.get("created_at"):
            order.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("submitted_at"):
            order.submitted_at = datetime.fromisoformat(data["submitted_at"])
        if data.get("filled_at"):
            order.filled_at = datetime.fromisoformat(data["filled_at"])

        return order
