"""Comprehensive risk management for weather trading.

Enforces strict position limits, loss limits, and trading criteria
to protect capital and ensure disciplined trading.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


class RiskStatus(str, Enum):
    """Risk check status."""
    OK = "OK"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


class StopReason(str, Enum):
    """Reason for trading stop."""
    NONE = "NONE"
    DAILY_LOSS = "DAILY_LOSS"
    WEEKLY_LOSS = "WEEKLY_LOSS"
    DRAWDOWN = "DRAWDOWN"
    MANUAL = "MANUAL"


@dataclass
class RiskCheck:
    """Result of a single risk check."""
    name: str
    status: RiskStatus
    message: str
    current_value: float
    limit_value: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == RiskStatus.OK

    @property
    def blocked(self) -> bool:
        return self.status == RiskStatus.BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "passed": self.passed,
            **self.details,
        }


@dataclass
class TradeRequest:
    """Request to execute a trade."""
    token_id: str
    condition_id: str  # Market ID
    outcome: str
    side: str  # "BUY" or "SELL"
    price: Decimal
    size: Decimal
    edge: float
    model_agreement: float
    liquidity: float
    city: str = ""
    target_date: date | None = None

    @property
    def notional_value(self) -> Decimal:
        """Total value of the trade."""
        return self.price * self.size


@dataclass
class Position:
    """An open position."""
    token_id: str
    condition_id: str
    outcome: str
    city: str
    target_date: date | None
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    opened_at: datetime

    @property
    def cost_basis(self) -> Decimal:
        return self.size * self.entry_price

    @property
    def market_value(self) -> Decimal:
        return self.size * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        return self.market_value - self.cost_basis

    @property
    def return_pct(self) -> float:
        if self.cost_basis > 0:
            return float(self.unrealized_pnl / self.cost_basis)
        return 0.0


@dataclass
class DailyPnL:
    """Daily P&L tracking."""
    date: date
    starting_bankroll: Decimal
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        if self.starting_bankroll > 0:
            return float(self.total_pnl / self.starting_bankroll)
        return 0.0

    @property
    def win_rate(self) -> float:
        if self.n_trades > 0:
            return self.n_wins / self.n_trades
        return 0.0


class RiskManager:
    """
    Comprehensive risk manager for weather trading.

    Enforces:
    - Position size limits (2% max per position)
    - Position count limits (5 concurrent, 3 per market)
    - Daily loss limit (5% → 24h stop)
    - Weekly loss limit (10% → 7 day stop)
    - Drawdown limit (20% → emergency stop)
    - Trade quality filters (edge, model agreement, liquidity)

    Usage:
        risk = RiskManager(starting_bankroll=Decimal("1500"))
        request = TradeRequest(...)

        can_trade, checks = risk.can_trade(request)
        if can_trade:
            # Execute trade
            risk.record_trade_open(position)
    """

    # Risk limits (can be overridden via constructor)
    MAX_POSITION_PCT = 0.02  # 2% of bankroll per position
    MAX_CONCURRENT_POSITIONS = 5
    MAX_POSITIONS_PER_MARKET = 3
    DAILY_LOSS_LIMIT_PCT = 0.05  # 5% daily loss → 24h stop
    WEEKLY_LOSS_LIMIT_PCT = 0.10  # 10% weekly loss → 7 day stop
    MAX_DRAWDOWN_PCT = 0.20  # 20% drawdown → emergency stop
    MIN_EDGE = 0.10  # 10% minimum edge
    MIN_MODEL_AGREEMENT = 0.65  # 65% model agreement
    MIN_LIQUIDITY = 10.0  # $10 minimum liquidity (weather markets have lower liquidity)

    def __init__(
        self,
        starting_bankroll: Decimal = Decimal("1500"),
        max_position_pct: float | None = None,
        max_concurrent_positions: int | None = None,
        max_positions_per_market: int | None = None,
        daily_loss_limit_pct: float | None = None,
        weekly_loss_limit_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        min_edge: float | None = None,
        min_model_agreement: float | None = None,
        min_liquidity: float | None = None,
    ):
        """
        Initialize the risk manager.

        Args:
            starting_bankroll: Initial bankroll in USD
            max_position_pct: Max position size as % of bankroll (default 2%)
            max_concurrent_positions: Max number of open positions (default 5)
            max_positions_per_market: Max positions per market (default 3)
            daily_loss_limit_pct: Daily loss limit % (default 5%)
            weekly_loss_limit_pct: Weekly loss limit % (default 10%)
            max_drawdown_pct: Max drawdown from peak % (default 20%)
            min_edge: Minimum edge to trade (default 10%)
            min_model_agreement: Minimum model agreement (default 65%)
            min_liquidity: Minimum liquidity in bucket (default $500)
        """
        # Bankroll tracking
        self._starting_bankroll = starting_bankroll
        self._current_bankroll = starting_bankroll
        self._peak_bankroll = starting_bankroll

        # Risk limits
        self.max_position_pct = max_position_pct or self.MAX_POSITION_PCT
        self.max_concurrent_positions = max_concurrent_positions or self.MAX_CONCURRENT_POSITIONS
        self.max_positions_per_market = max_positions_per_market or self.MAX_POSITIONS_PER_MARKET
        self.daily_loss_limit_pct = daily_loss_limit_pct or self.DAILY_LOSS_LIMIT_PCT
        self.weekly_loss_limit_pct = weekly_loss_limit_pct or self.WEEKLY_LOSS_LIMIT_PCT
        self.max_drawdown_pct = max_drawdown_pct or self.MAX_DRAWDOWN_PCT
        self.min_edge = min_edge or self.MIN_EDGE
        self.min_model_agreement = min_model_agreement or self.MIN_MODEL_AGREEMENT
        self.min_liquidity = min_liquidity or self.MIN_LIQUIDITY

        # Position tracking
        self._positions: dict[str, Position] = {}  # token_id -> Position

        # P&L tracking
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._total_realized_pnl = Decimal("0")

        # Stop flags
        self._stop_reason = StopReason.NONE
        self._stop_until: datetime | None = None
        self._emergency_stopped = False

        logger.info(
            "Risk manager initialized",
            starting_bankroll=str(starting_bankroll),
            max_position_pct=self.max_position_pct,
            max_concurrent_positions=self.max_concurrent_positions,
            daily_loss_limit_pct=self.daily_loss_limit_pct,
            max_drawdown_pct=self.max_drawdown_pct,
        )

    # =========================================================================
    # Core Trading Check
    # =========================================================================

    def can_trade(self, request: TradeRequest) -> tuple[bool, list[RiskCheck]]:
        """
        Check if a trade is allowed.

        Runs ALL risk checks and returns whether the trade can proceed.

        Args:
            request: The trade request to evaluate

        Returns:
            Tuple of (can_trade: bool, checks: list[RiskCheck])
        """
        checks: list[RiskCheck] = []

        # 1. Emergency stop check
        checks.append(self._check_emergency_stop())

        # 2. Trading stop check (daily/weekly loss)
        checks.append(self._check_trading_stop())

        # 3. Drawdown check
        checks.append(self._check_drawdown())

        # 4. Daily loss limit check
        checks.append(self._check_daily_loss())

        # 5. Weekly loss limit check
        checks.append(self._check_weekly_loss())

        # 6. Position size check
        checks.append(self._check_position_size(request))

        # 7. Concurrent positions check
        checks.append(self._check_concurrent_positions())

        # 8. Positions per market check
        checks.append(self._check_positions_per_market(request.condition_id))

        # 9. Edge check
        checks.append(self._check_edge(request.edge))

        # 10. Model agreement check
        checks.append(self._check_model_agreement(request.model_agreement))

        # 11. Liquidity check
        checks.append(self._check_liquidity(request.liquidity))

        # Determine if trade is allowed
        can_trade = all(check.passed or check.status == RiskStatus.WARNING for check in checks)
        blocked_checks = [c for c in checks if c.blocked]

        if blocked_checks:
            logger.warning(
                "Trade blocked",
                token_id=request.token_id,
                outcome=request.outcome,
                blocked_by=[c.name for c in blocked_checks],
            )
        else:
            logger.debug(
                "Trade allowed",
                token_id=request.token_id,
                outcome=request.outcome,
                edge=request.edge,
            )

        return can_trade, checks

    # =========================================================================
    # Individual Risk Checks
    # =========================================================================

    def _check_emergency_stop(self) -> RiskCheck:
        """Check if emergency stop is active."""
        if self._emergency_stopped:
            return RiskCheck(
                name="emergency_stop",
                status=RiskStatus.BLOCKED,
                message="Emergency stop active - manual restart required",
                current_value=1.0,
                limit_value=0.0,
                details={"reason": self._stop_reason.value},
            )
        return RiskCheck(
            name="emergency_stop",
            status=RiskStatus.OK,
            message="No emergency stop",
            current_value=0.0,
            limit_value=1.0,
        )

    def _check_trading_stop(self) -> RiskCheck:
        """Check if trading is temporarily stopped (daily/weekly loss)."""
        if self._stop_until and datetime.utcnow() < self._stop_until:
            remaining = self._stop_until - datetime.utcnow()
            hours_remaining = remaining.total_seconds() / 3600
            return RiskCheck(
                name="trading_stop",
                status=RiskStatus.BLOCKED,
                message=f"Trading stopped for {hours_remaining:.1f} more hours",
                current_value=hours_remaining,
                limit_value=0.0,
                details={
                    "reason": self._stop_reason.value,
                    "stop_until": self._stop_until.isoformat(),
                },
            )
        return RiskCheck(
            name="trading_stop",
            status=RiskStatus.OK,
            message="Trading allowed",
            current_value=0.0,
            limit_value=1.0,
        )

    def _check_drawdown(self) -> RiskCheck:
        """Check drawdown from peak (20% limit → emergency stop)."""
        drawdown = self.current_drawdown

        if drawdown >= self.max_drawdown_pct:
            self._trigger_emergency_stop(StopReason.DRAWDOWN)
            return RiskCheck(
                name="drawdown",
                status=RiskStatus.BLOCKED,
                message=f"Drawdown {drawdown:.1%} exceeds {self.max_drawdown_pct:.0%} limit",
                current_value=drawdown,
                limit_value=self.max_drawdown_pct,
                details={
                    "peak_bankroll": str(self._peak_bankroll),
                    "current_bankroll": str(self._current_bankroll),
                },
            )

        # Warning at 80% of limit
        if drawdown >= self.max_drawdown_pct * 0.8:
            return RiskCheck(
                name="drawdown",
                status=RiskStatus.WARNING,
                message=f"Drawdown {drawdown:.1%} approaching limit",
                current_value=drawdown,
                limit_value=self.max_drawdown_pct,
            )

        return RiskCheck(
            name="drawdown",
            status=RiskStatus.OK,
            message=f"Drawdown {drawdown:.1%} within limits",
            current_value=drawdown,
            limit_value=self.max_drawdown_pct,
        )

    def _check_daily_loss(self) -> RiskCheck:
        """Check daily loss (5% limit → 24h stop)."""
        daily_pnl = self._get_today_pnl()
        loss_pct = -daily_pnl.pnl_pct if daily_pnl.pnl_pct < 0 else 0

        if loss_pct >= self.daily_loss_limit_pct:
            self._trigger_trading_stop(StopReason.DAILY_LOSS, hours=24)
            return RiskCheck(
                name="daily_loss",
                status=RiskStatus.BLOCKED,
                message=f"Daily loss {loss_pct:.1%} exceeds {self.daily_loss_limit_pct:.0%} limit",
                current_value=loss_pct,
                limit_value=self.daily_loss_limit_pct,
                details={
                    "realized_pnl": str(daily_pnl.realized_pnl),
                    "unrealized_pnl": str(daily_pnl.unrealized_pnl),
                },
            )

        # Warning at 80% of limit
        if loss_pct >= self.daily_loss_limit_pct * 0.8:
            return RiskCheck(
                name="daily_loss",
                status=RiskStatus.WARNING,
                message=f"Daily loss {loss_pct:.1%} approaching limit",
                current_value=loss_pct,
                limit_value=self.daily_loss_limit_pct,
            )

        return RiskCheck(
            name="daily_loss",
            status=RiskStatus.OK,
            message=f"Daily loss {loss_pct:.1%} within limits",
            current_value=loss_pct,
            limit_value=self.daily_loss_limit_pct,
        )

    def _check_weekly_loss(self) -> RiskCheck:
        """Check weekly loss (10% limit → 7 day stop)."""
        weekly_pnl = self._get_weekly_pnl()
        loss_pct = -weekly_pnl if weekly_pnl < 0 else 0

        if loss_pct >= self.weekly_loss_limit_pct:
            self._trigger_trading_stop(StopReason.WEEKLY_LOSS, hours=168)  # 7 days
            return RiskCheck(
                name="weekly_loss",
                status=RiskStatus.BLOCKED,
                message=f"Weekly loss {loss_pct:.1%} exceeds {self.weekly_loss_limit_pct:.0%} limit",
                current_value=loss_pct,
                limit_value=self.weekly_loss_limit_pct,
            )

        # Warning at 80% of limit
        if loss_pct >= self.weekly_loss_limit_pct * 0.8:
            return RiskCheck(
                name="weekly_loss",
                status=RiskStatus.WARNING,
                message=f"Weekly loss {loss_pct:.1%} approaching limit",
                current_value=loss_pct,
                limit_value=self.weekly_loss_limit_pct,
            )

        return RiskCheck(
            name="weekly_loss",
            status=RiskStatus.OK,
            message=f"Weekly loss {loss_pct:.1%} within limits",
            current_value=loss_pct,
            limit_value=self.weekly_loss_limit_pct,
        )

    def _check_position_size(self, request: TradeRequest) -> RiskCheck:
        """Check position size (2% max of bankroll)."""
        max_size = float(self._current_bankroll) * self.max_position_pct
        requested_size = float(request.notional_value)
        size_pct = requested_size / float(self._current_bankroll) if self._current_bankroll > 0 else 0

        if requested_size > max_size:
            return RiskCheck(
                name="position_size",
                status=RiskStatus.BLOCKED,
                message=f"Position ${requested_size:.2f} exceeds ${max_size:.2f} limit ({self.max_position_pct:.0%})",
                current_value=size_pct,
                limit_value=self.max_position_pct,
                details={
                    "requested_size": requested_size,
                    "max_size": max_size,
                },
            )

        # Warning at 80% of limit
        if size_pct >= self.max_position_pct * 0.8:
            return RiskCheck(
                name="position_size",
                status=RiskStatus.WARNING,
                message=f"Position size {size_pct:.1%} near limit",
                current_value=size_pct,
                limit_value=self.max_position_pct,
            )

        return RiskCheck(
            name="position_size",
            status=RiskStatus.OK,
            message=f"Position size {size_pct:.1%} within limits",
            current_value=size_pct,
            limit_value=self.max_position_pct,
        )

    def _check_concurrent_positions(self) -> RiskCheck:
        """Check concurrent positions (5 max)."""
        n_positions = len(self._positions)

        if n_positions >= self.max_concurrent_positions:
            return RiskCheck(
                name="concurrent_positions",
                status=RiskStatus.BLOCKED,
                message=f"At max positions ({n_positions}/{self.max_concurrent_positions})",
                current_value=n_positions,
                limit_value=self.max_concurrent_positions,
            )

        # Warning at 80% of limit
        if n_positions >= self.max_concurrent_positions * 0.8:
            return RiskCheck(
                name="concurrent_positions",
                status=RiskStatus.WARNING,
                message=f"Approaching max positions ({n_positions}/{self.max_concurrent_positions})",
                current_value=n_positions,
                limit_value=self.max_concurrent_positions,
            )

        return RiskCheck(
            name="concurrent_positions",
            status=RiskStatus.OK,
            message=f"Positions: {n_positions}/{self.max_concurrent_positions}",
            current_value=n_positions,
            limit_value=self.max_concurrent_positions,
        )

    def _check_positions_per_market(self, condition_id: str) -> RiskCheck:
        """Check positions per market (3 max)."""
        n_in_market = sum(
            1 for p in self._positions.values()
            if p.condition_id == condition_id
        )

        if n_in_market >= self.max_positions_per_market:
            return RiskCheck(
                name="positions_per_market",
                status=RiskStatus.BLOCKED,
                message=f"At max positions in market ({n_in_market}/{self.max_positions_per_market})",
                current_value=n_in_market,
                limit_value=self.max_positions_per_market,
                details={"condition_id": condition_id},
            )

        return RiskCheck(
            name="positions_per_market",
            status=RiskStatus.OK,
            message=f"Positions in market: {n_in_market}/{self.max_positions_per_market}",
            current_value=n_in_market,
            limit_value=self.max_positions_per_market,
        )

    def _check_edge(self, edge: float) -> RiskCheck:
        """Check minimum edge (10% required)."""
        abs_edge = abs(edge)

        if abs_edge < self.min_edge:
            return RiskCheck(
                name="edge",
                status=RiskStatus.BLOCKED,
                message=f"Edge {abs_edge:.1%} below {self.min_edge:.0%} minimum",
                current_value=abs_edge,
                limit_value=self.min_edge,
            )

        return RiskCheck(
            name="edge",
            status=RiskStatus.OK,
            message=f"Edge {abs_edge:.1%} meets minimum",
            current_value=abs_edge,
            limit_value=self.min_edge,
        )

    def _check_model_agreement(self, agreement: float) -> RiskCheck:
        """Check model agreement (65% required)."""
        if agreement < self.min_model_agreement:
            return RiskCheck(
                name="model_agreement",
                status=RiskStatus.BLOCKED,
                message=f"Model agreement {agreement:.1%} below {self.min_model_agreement:.0%} minimum",
                current_value=agreement,
                limit_value=self.min_model_agreement,
            )

        return RiskCheck(
            name="model_agreement",
            status=RiskStatus.OK,
            message=f"Model agreement {agreement:.1%} meets minimum",
            current_value=agreement,
            limit_value=self.min_model_agreement,
        )

    def _check_liquidity(self, liquidity: float) -> RiskCheck:
        """Check bucket liquidity ($500 required)."""
        if liquidity < self.min_liquidity:
            return RiskCheck(
                name="liquidity",
                status=RiskStatus.BLOCKED,
                message=f"Liquidity ${liquidity:.0f} below ${self.min_liquidity:.0f} minimum",
                current_value=liquidity,
                limit_value=self.min_liquidity,
            )

        # Warning if liquidity is low but acceptable
        if liquidity < self.min_liquidity * 2:
            return RiskCheck(
                name="liquidity",
                status=RiskStatus.WARNING,
                message=f"Low liquidity: ${liquidity:.0f}",
                current_value=liquidity,
                limit_value=self.min_liquidity,
            )

        return RiskCheck(
            name="liquidity",
            status=RiskStatus.OK,
            message=f"Liquidity ${liquidity:.0f} adequate",
            current_value=liquidity,
            limit_value=self.min_liquidity,
        )

    # =========================================================================
    # Position Management
    # =========================================================================

    def has_position(self, token_id: str) -> bool:
        """Check if we already have a position in this token."""
        return token_id in self._positions

    def record_trade_open(self, position: Position) -> None:
        """Record opening a new position."""
        self._positions[position.token_id] = position

        # Update daily P&L tracking
        daily = self._get_today_pnl()
        daily.n_trades += 1

        logger.info(
            "Position opened",
            token_id=position.token_id,
            outcome=position.outcome,
            size=str(position.size),
            entry_price=str(position.entry_price),
        )

    def record_trade_close(
        self,
        token_id: str,
        exit_price: Decimal,
        realized_pnl: Decimal,
    ) -> None:
        """Record closing a position."""
        position = self._positions.pop(token_id, None)

        if position is None:
            logger.warning(f"Attempted to close unknown position: {token_id}")
            return

        # Update P&L tracking
        self._total_realized_pnl += realized_pnl
        daily = self._get_today_pnl()
        daily.realized_pnl += realized_pnl
        daily.n_trades += 1

        if realized_pnl > 0:
            daily.n_wins += 1
        elif realized_pnl < 0:
            daily.n_losses += 1

        # Update bankroll
        self._current_bankroll += realized_pnl
        if self._current_bankroll > self._peak_bankroll:
            self._peak_bankroll = self._current_bankroll

        logger.info(
            "Position closed",
            token_id=token_id,
            outcome=position.outcome if position else "unknown",
            exit_price=str(exit_price),
            realized_pnl=str(realized_pnl),
        )

    def update_position_price(self, token_id: str, current_price: Decimal) -> None:
        """Update a position's current market price."""
        if token_id in self._positions:
            self._positions[token_id].current_price = current_price

    def update_unrealized_pnl(self) -> Decimal:
        """Calculate and update unrealized P&L across all positions."""
        total_unrealized = Decimal("0")
        for position in self._positions.values():
            total_unrealized += position.unrealized_pnl

        daily = self._get_today_pnl()
        daily.unrealized_pnl = total_unrealized

        return total_unrealized

    # =========================================================================
    # Stop Management
    # =========================================================================

    def _trigger_emergency_stop(self, reason: StopReason) -> None:
        """Trigger emergency stop - requires manual restart."""
        self._emergency_stopped = True
        self._stop_reason = reason

        logger.critical(
            "EMERGENCY STOP TRIGGERED",
            reason=reason.value,
            drawdown=self.current_drawdown,
            current_bankroll=str(self._current_bankroll),
            peak_bankroll=str(self._peak_bankroll),
        )

    def _trigger_trading_stop(self, reason: StopReason, hours: int) -> None:
        """Trigger temporary trading stop."""
        self._stop_reason = reason
        self._stop_until = datetime.utcnow() + timedelta(hours=hours)

        logger.warning(
            "Trading stop triggered",
            reason=reason.value,
            stop_until=self._stop_until.isoformat(),
            hours=hours,
        )

    def reset_emergency_stop(self, confirm: bool = False) -> bool:
        """
        Reset emergency stop - requires explicit confirmation.

        Args:
            confirm: Must be True to actually reset

        Returns:
            True if reset was successful
        """
        if not confirm:
            logger.warning("Emergency stop reset requires confirm=True")
            return False

        if not self._emergency_stopped:
            return True

        self._emergency_stopped = False
        self._stop_reason = StopReason.NONE
        self._stop_until = None

        logger.info("Emergency stop reset by user")
        return True

    def clear_trading_stop(self) -> None:
        """Clear temporary trading stop (for testing/override)."""
        self._stop_reason = StopReason.NONE
        self._stop_until = None
        logger.info("Trading stop cleared")

    # =========================================================================
    # P&L Tracking
    # =========================================================================

    def _get_today_pnl(self) -> DailyPnL:
        """Get or create today's P&L record."""
        today = date.today()
        if today not in self._daily_pnl:
            self._daily_pnl[today] = DailyPnL(
                date=today,
                starting_bankroll=self._current_bankroll,
            )
        return self._daily_pnl[today]

    def _get_weekly_pnl(self) -> float:
        """Calculate P&L percentage over the last 7 days."""
        today = date.today()
        week_ago = today - timedelta(days=7)

        total_pnl = Decimal("0")
        starting_bankroll = self._starting_bankroll

        for d, daily in self._daily_pnl.items():
            if d >= week_ago:
                total_pnl += daily.total_pnl
                if d == week_ago:
                    starting_bankroll = daily.starting_bankroll

        if starting_bankroll > 0:
            return float(total_pnl / starting_bankroll)
        return 0.0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def current_bankroll(self) -> Decimal:
        """Current bankroll value."""
        return self._current_bankroll

    @property
    def peak_bankroll(self) -> Decimal:
        """Peak bankroll (high-water mark)."""
        return self._peak_bankroll

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self._peak_bankroll > 0:
            return float((self._peak_bankroll - self._current_bankroll) / self._peak_bankroll)
        return 0.0

    @property
    def n_positions(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    @property
    def total_position_value(self) -> Decimal:
        """Total value of all positions."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def is_trading_allowed(self) -> bool:
        """Quick check if trading is currently allowed."""
        if self._emergency_stopped:
            return False
        if self._stop_until and datetime.utcnow() < self._stop_until:
            return False
        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def calculate_position_size(
        self,
        edge: float,
        price: Decimal,
        kelly_fraction: float = 0.5,
    ) -> Decimal:
        """
        Calculate recommended position size using Half-Kelly.

        Args:
            edge: Expected edge as decimal
            price: Current price
            kelly_fraction: Fraction of Kelly to use (default 0.5)

        Returns:
            Recommended position size in USD
        """
        if edge <= 0 or float(price) <= 0 or float(price) >= 1:
            return Decimal("0")

        # Kelly formula
        p = float(price) + edge * float(price)  # Fair probability
        b = (1 - float(price)) / float(price)  # Odds
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%

        # Apply fraction and position limit
        fraction = kelly * kelly_fraction
        max_pct = self.max_position_pct

        size = float(self._current_bankroll) * min(fraction, max_pct)

        return Decimal(str(round(size, 2)))

    def get_status(self) -> dict[str, Any]:
        """Get current risk status summary."""
        daily = self._get_today_pnl()

        return {
            "bankroll": {
                "starting": str(self._starting_bankroll),
                "current": str(self._current_bankroll),
                "peak": str(self._peak_bankroll),
                "drawdown_pct": round(self.current_drawdown * 100, 2),
            },
            "positions": {
                "count": self.n_positions,
                "max": self.max_concurrent_positions,
                "total_value": str(self.total_position_value),
                "unrealized_pnl": str(self.total_unrealized_pnl),
            },
            "daily": {
                "date": str(daily.date),
                "realized_pnl": str(daily.realized_pnl),
                "unrealized_pnl": str(daily.unrealized_pnl),
                "total_pnl_pct": round(daily.pnl_pct * 100, 2),
                "trades": daily.n_trades,
                "win_rate": round(daily.win_rate * 100, 1),
            },
            "stops": {
                "emergency_stopped": self._emergency_stopped,
                "stop_reason": self._stop_reason.value,
                "stop_until": self._stop_until.isoformat() if self._stop_until else None,
                "trading_allowed": self.is_trading_allowed,
            },
            "limits": {
                "max_position_pct": self.max_position_pct,
                "daily_loss_limit_pct": self.daily_loss_limit_pct,
                "weekly_loss_limit_pct": self.weekly_loss_limit_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "min_edge": self.min_edge,
                "min_model_agreement": self.min_model_agreement,
                "min_liquidity": self.min_liquidity,
            },
        }

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())


def demo_risk_manager():
    """Demonstrate the risk manager."""
    print(f"\n{'='*60}")
    print("Risk Manager Demo")
    print(f"{'='*60}\n")

    # Initialize with $1500 bankroll
    risk = RiskManager(starting_bankroll=Decimal("1500"))

    print("Initial Status:")
    status = risk.get_status()
    print(f"  Bankroll: ${status['bankroll']['current']}")
    print(f"  Max Position: {risk.max_position_pct:.0%} = ${float(risk.current_bankroll) * risk.max_position_pct:.2f}")
    print(f"  Trading Allowed: {status['stops']['trading_allowed']}")

    # Test trade requests
    print(f"\n{'='*60}")
    print("Testing Trade Requests")
    print(f"{'='*60}\n")

    # Good trade
    request1 = TradeRequest(
        token_id="token_1",
        condition_id="market_1",
        outcome="55-60°F",
        side="BUY",
        price=Decimal("0.25"),
        size=Decimal("20"),  # $5 notional (0.3% of bankroll)
        edge=0.15,  # 15% edge
        model_agreement=0.75,
        liquidity=1000.0,
    )

    can_trade, checks = risk.can_trade(request1)
    print(f"Trade 1: {request1.outcome} @ ${request1.price}")
    print(f"  Can Trade: {can_trade}")
    for check in checks:
        if not check.passed:
            print(f"  - {check.name}: {check.status.value} - {check.message}")

    # Trade with insufficient edge
    request2 = TradeRequest(
        token_id="token_2",
        condition_id="market_1",
        outcome="60-65°F",
        side="BUY",
        price=Decimal("0.30"),
        size=Decimal("10"),
        edge=0.05,  # Only 5% edge
        model_agreement=0.75,
        liquidity=800.0,
    )

    can_trade, checks = risk.can_trade(request2)
    print(f"\nTrade 2: {request2.outcome} @ ${request2.price} (low edge)")
    print(f"  Can Trade: {can_trade}")
    blocked = [c for c in checks if c.blocked]
    for check in blocked:
        print(f"  - BLOCKED: {check.name} - {check.message}")

    # Trade with low liquidity
    request3 = TradeRequest(
        token_id="token_3",
        condition_id="market_1",
        outcome="50-55°F",
        side="BUY",
        price=Decimal("0.20"),
        size=Decimal("15"),
        edge=0.20,
        model_agreement=0.70,
        liquidity=200.0,  # Below $500 minimum
    )

    can_trade, checks = risk.can_trade(request3)
    print(f"\nTrade 3: {request3.outcome} @ ${request3.price} (low liquidity)")
    print(f"  Can Trade: {can_trade}")
    blocked = [c for c in checks if c.blocked]
    for check in blocked:
        print(f"  - BLOCKED: {check.name} - {check.message}")

    # Trade too large
    request4 = TradeRequest(
        token_id="token_4",
        condition_id="market_2",
        outcome="70-75°F",
        side="BUY",
        price=Decimal("0.40"),
        size=Decimal("100"),  # $40 notional (2.7% of bankroll)
        edge=0.25,
        model_agreement=0.80,
        liquidity=2000.0,
    )

    can_trade, checks = risk.can_trade(request4)
    print(f"\nTrade 4: {request4.outcome} @ ${request4.price} (size too large)")
    print(f"  Can Trade: {can_trade}")
    blocked = [c for c in checks if c.blocked]
    for check in blocked:
        print(f"  - BLOCKED: {check.name} - {check.message}")

    # Simulate opening positions
    print(f"\n{'='*60}")
    print("Simulating Position Management")
    print(f"{'='*60}\n")

    for i in range(5):
        pos = Position(
            token_id=f"pos_{i}",
            condition_id="market_1",
            outcome=f"Bucket {i}",
            city="Dallas",
            target_date=date.today(),
            size=Decimal("10"),
            entry_price=Decimal("0.25"),
            current_price=Decimal("0.25"),
            opened_at=datetime.utcnow(),
        )
        risk.record_trade_open(pos)

    print(f"Opened 5 positions")
    print(f"  Position count: {risk.n_positions}")

    # Try to open 6th position
    can_trade, checks = risk.can_trade(request1)
    print(f"\nTrying to open 6th position:")
    print(f"  Can Trade: {can_trade}")
    blocked = [c for c in checks if c.blocked]
    for check in blocked:
        print(f"  - BLOCKED: {check.name} - {check.message}")

    # Simulate drawdown
    print(f"\n{'='*60}")
    print("Simulating Drawdown Scenario")
    print(f"{'='*60}\n")

    # Close positions with losses
    for i in range(5):
        risk.record_trade_close(
            token_id=f"pos_{i}",
            exit_price=Decimal("0.10"),
            realized_pnl=Decimal("-1.50"),  # -$1.50 each
        )

    print(f"Closed all positions with losses")
    print(f"  Current bankroll: ${risk.current_bankroll}")
    print(f"  Drawdown: {risk.current_drawdown:.1%}")

    # Check if we can still trade
    can_trade, checks = risk.can_trade(request1)
    print(f"\nCan we still trade?")
    print(f"  Trading allowed: {risk.is_trading_allowed}")

    print(f"\n{'='*60}")
    print("Final Status")
    print(f"{'='*60}")

    status = risk.get_status()
    print(f"\nBankroll: ${status['bankroll']['current']} (peak: ${status['bankroll']['peak']})")
    print(f"Drawdown: {status['bankroll']['drawdown_pct']}%")
    print(f"Daily P&L: {status['daily']['total_pnl_pct']}%")
    print(f"Trading Allowed: {status['stops']['trading_allowed']}")

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    demo_risk_manager()
