"""Position tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
import uuid


@dataclass
class Position:
    """
    A trading position.

    Tracks holdings in a specific market outcome.
    """

    id: str = field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:12]}")
    token_id: str = ""
    condition_id: str = ""
    bucket_outcome: str = ""
    city: str = ""
    target_date: str = ""

    # Position details
    size: Decimal = Decimal("0")  # Number of shares
    avg_entry_price: Decimal = Decimal("0")  # Average entry price
    current_price: Decimal = Decimal("0")  # Current market price
    opened_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # P&L
    realized_pnl: Decimal = Decimal("0")

    @property
    def cost_basis(self) -> Decimal:
        """Total cost of position."""
        return self.size * self.avg_entry_price

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized P&L."""
        return self.market_value - self.cost_basis

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.cost_basis > 0:
            return float(self.unrealized_pnl / self.cost_basis)
        return 0.0

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.size > 0

    def add_to_position(self, size: Decimal, price: Decimal) -> None:
        """Add to existing position."""
        # Update weighted average entry price
        total_cost = self.cost_basis + size * price
        self.size += size
        if self.size > 0:
            self.avg_entry_price = total_cost / self.size
        self.last_updated = datetime.utcnow()

    def reduce_position(self, size: Decimal, price: Decimal) -> Decimal:
        """
        Reduce position and return realized P&L.

        Args:
            size: Number of shares to sell
            price: Sale price

        Returns:
            Realized P&L from this sale
        """
        if size > self.size:
            size = self.size

        # Calculate realized P&L for this portion
        cost_of_sold = size * self.avg_entry_price
        proceeds = size * price
        realized = proceeds - cost_of_sold

        self.realized_pnl += realized
        self.size -= size
        self.last_updated = datetime.utcnow()

        return realized

    def update_price(self, price: Decimal) -> None:
        """Update current market price."""
        self.current_price = price
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "bucket_outcome": self.bucket_outcome,
            "city": self.city,
            "target_date": self.target_date,
            "size": str(self.size),
            "avg_entry_price": str(self.avg_entry_price),
            "current_price": str(self.current_price),
            "opened_at": self.opened_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "cost_basis": str(self.cost_basis),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "return_pct": self.return_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        """Create from dictionary."""
        pos = cls(
            id=data.get("id", ""),
            token_id=data.get("token_id", ""),
            condition_id=data.get("condition_id", ""),
            bucket_outcome=data.get("bucket_outcome", ""),
            city=data.get("city", ""),
            target_date=data.get("target_date", ""),
            size=Decimal(data.get("size", "0")),
            avg_entry_price=Decimal(data.get("avg_entry_price", "0")),
            current_price=Decimal(data.get("current_price", "0")),
            realized_pnl=Decimal(data.get("realized_pnl", "0")),
        )

        if data.get("opened_at"):
            pos.opened_at = datetime.fromisoformat(data["opened_at"])
        if data.get("last_updated"):
            pos.last_updated = datetime.fromisoformat(data["last_updated"])

        return pos


class PositionTracker:
    """
    Tracks all open positions.

    Provides aggregation and lookup functions.
    """

    def __init__(self):
        """Initialize the tracker."""
        self._positions: dict[str, Position] = {}  # token_id -> Position

    def get_position(self, token_id: str) -> Position | None:
        """Get position for a token."""
        return self._positions.get(token_id)

    def get_or_create_position(
        self,
        token_id: str,
        condition_id: str = "",
        bucket_outcome: str = "",
        city: str = "",
        target_date: str = "",
    ) -> Position:
        """Get or create a position."""
        if token_id not in self._positions:
            self._positions[token_id] = Position(
                token_id=token_id,
                condition_id=condition_id,
                bucket_outcome=bucket_outcome,
                city=city,
                target_date=target_date,
            )
        return self._positions[token_id]

    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self._positions.values() if p.is_open]

    def get_positions_by_market(self, condition_id: str) -> list[Position]:
        """Get positions for a specific market."""
        return [
            p for p in self._positions.values()
            if p.condition_id == condition_id and p.is_open
        ]

    def get_positions_by_city(self, city: str) -> list[Position]:
        """Get positions for a specific city."""
        return [
            p for p in self._positions.values()
            if p.city == city and p.is_open
        ]

    @property
    def total_value(self) -> Decimal:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.get_open_positions())

    @property
    def total_cost(self) -> Decimal:
        """Total cost basis of all positions."""
        return sum(p.cost_basis for p in self.get_open_positions())

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.get_open_positions())

    @property
    def total_realized_pnl(self) -> Decimal:
        """Total realized P&L."""
        return sum(p.realized_pnl for p in self._positions.values())

    def remove_closed_positions(self) -> None:
        """Remove positions with zero size."""
        self._positions = {
            k: v for k, v in self._positions.items() if v.is_open
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "positions": [p.to_dict() for p in self._positions.values()],
            "total_value": str(self.total_value),
            "total_cost": str(self.total_cost),
            "total_unrealized_pnl": str(self.total_unrealized_pnl),
            "total_realized_pnl": str(self.total_realized_pnl),
            "n_open": len(self.get_open_positions()),
        }

    def load_positions(self, positions: list[Position]) -> None:
        """Load positions from list."""
        for pos in positions:
            self._positions[pos.token_id] = pos
