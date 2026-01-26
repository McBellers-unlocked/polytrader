"""Risk management and position limits."""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from src.config import get_settings
from src.strategy.edge import TradeSignal
from src.logging import get_logger

logger = get_logger(__name__)


class RiskStatus(str, Enum):
    """Risk check status."""

    OK = "OK"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


@dataclass
class RiskCheck:
    """Result of a risk check."""

    name: str
    status: RiskStatus
    message: str
    current_value: float
    limit_value: float

    @property
    def is_ok(self) -> bool:
        """Check if risk check passed."""
        return self.status == RiskStatus.OK

    @property
    def is_blocked(self) -> bool:
        """Check if risk check blocks the trade."""
        return self.status == RiskStatus.BLOCKED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
        }


@dataclass
class DailyPnL:
    """Daily profit and loss tracking."""

    date: date
    starting_bankroll: Decimal
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    trades: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L for the day."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of starting bankroll."""
        if self.starting_bankroll > 0:
            return float(self.total_pnl / self.starting_bankroll)
        return 0.0

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.trades > 0:
            return self.wins / self.trades
        return 0.0


class RiskManager:
    """
    Manages trading risk and position limits.

    Implements:
    - Per-position size limits (2% max)
    - Daily loss stop (5%)
    - Maximum drawdown emergency stop (20%)
    - Model agreement filter
    - Liquidity checks
    """

    def __init__(self):
        """Initialize the risk manager."""
        self.settings = get_settings()
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._peak_bankroll: Decimal = Decimal(str(self.settings.starting_bankroll))
        self._current_bankroll: Decimal = Decimal(str(self.settings.starting_bankroll))
        self._positions: dict[str, Decimal] = {}  # token_id -> position size
        self._is_emergency_stopped: bool = False

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self._peak_bankroll > 0:
            return float(
                (self._peak_bankroll - self._current_bankroll) / self._peak_bankroll
            )
        return 0.0

    @property
    def today_pnl(self) -> DailyPnL:
        """Get today's P&L."""
        today = date.today()
        if today not in self._daily_pnl:
            self._daily_pnl[today] = DailyPnL(
                date=today,
                starting_bankroll=self._current_bankroll,
            )
        return self._daily_pnl[today]

    @property
    def total_position_size(self) -> Decimal:
        """Total capital in positions."""
        return sum(self._positions.values(), Decimal("0"))

    def update_bankroll(self, new_bankroll: Decimal) -> None:
        """Update current bankroll and peak."""
        self._current_bankroll = new_bankroll
        if new_bankroll > self._peak_bankroll:
            self._peak_bankroll = new_bankroll

    def record_trade(
        self,
        token_id: str,
        size: Decimal,
        pnl: Decimal | None = None,
    ) -> None:
        """
        Record a trade for tracking.

        Args:
            token_id: The token traded
            size: Position size
            pnl: Realized P&L if closing position
        """
        self._positions[token_id] = self._positions.get(token_id, Decimal("0")) + size

        if pnl is not None:
            self.today_pnl.realized_pnl += pnl
            self.today_pnl.trades += 1
            if pnl > 0:
                self.today_pnl.wins += 1
            elif pnl < 0:
                self.today_pnl.losses += 1

    def check_trade(
        self,
        signal: TradeSignal,
    ) -> list[RiskCheck]:
        """
        Run all risk checks for a trade signal.

        Args:
            signal: The trade signal to check

        Returns:
            List of RiskCheck results
        """
        checks: list[RiskCheck] = []

        # Emergency stop check
        checks.append(self._check_emergency_stop())

        # Position size check
        checks.append(self._check_position_size(signal.suggested_size))

        # Daily loss check
        checks.append(self._check_daily_loss())

        # Drawdown check
        checks.append(self._check_drawdown())

        # Model agreement check
        checks.append(self._check_model_agreement(signal.model_agreement))

        # Liquidity check
        checks.append(self._check_liquidity(signal))

        # Log summary
        blocked = [c for c in checks if c.is_blocked]
        warnings = [c for c in checks if c.status == RiskStatus.WARNING]

        if blocked:
            logger.warning(
                "Trade blocked by risk checks",
                signal=signal.bucket.outcome,
                blocked=[c.name for c in blocked],
            )
        elif warnings:
            logger.info(
                "Trade passed with warnings",
                signal=signal.bucket.outcome,
                warnings=[c.name for c in warnings],
            )

        return checks

    def is_trade_allowed(self, checks: list[RiskCheck]) -> bool:
        """Check if a trade is allowed based on risk checks."""
        return not any(c.is_blocked for c in checks)

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop."""
        self._is_emergency_stopped = True
        logger.critical(
            "Emergency stop triggered",
            reason=reason,
            drawdown=self.current_drawdown,
            daily_pnl=float(self.today_pnl.total_pnl),
        )

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (manual override)."""
        self._is_emergency_stopped = False
        logger.info("Emergency stop reset")

    def _check_emergency_stop(self) -> RiskCheck:
        """Check if emergency stop is active."""
        if self._is_emergency_stopped:
            return RiskCheck(
                name="emergency_stop",
                status=RiskStatus.BLOCKED,
                message="Emergency stop is active",
                current_value=1.0,
                limit_value=0.0,
            )
        return RiskCheck(
            name="emergency_stop",
            status=RiskStatus.OK,
            message="No emergency stop",
            current_value=0.0,
            limit_value=1.0,
        )

    def _check_position_size(self, size: Decimal) -> RiskCheck:
        """Check if position size is within limits."""
        max_position = self._current_bankroll * Decimal(
            str(self.settings.max_position_pct)
        )
        size_pct = float(size / self._current_bankroll) if self._current_bankroll > 0 else 0

        if size > max_position:
            return RiskCheck(
                name="position_size",
                status=RiskStatus.BLOCKED,
                message=f"Position size {size} exceeds max {max_position}",
                current_value=size_pct,
                limit_value=self.settings.max_position_pct,
            )

        if size_pct > self.settings.max_position_pct * 0.8:
            return RiskCheck(
                name="position_size",
                status=RiskStatus.WARNING,
                message=f"Position size near limit",
                current_value=size_pct,
                limit_value=self.settings.max_position_pct,
            )

        return RiskCheck(
            name="position_size",
            status=RiskStatus.OK,
            message="Position size OK",
            current_value=size_pct,
            limit_value=self.settings.max_position_pct,
        )

    def _check_daily_loss(self) -> RiskCheck:
        """Check daily loss limit."""
        daily_pnl_pct = self.today_pnl.pnl_pct

        if daily_pnl_pct < -self.settings.daily_loss_stop_pct:
            self.trigger_emergency_stop("Daily loss limit exceeded")
            return RiskCheck(
                name="daily_loss",
                status=RiskStatus.BLOCKED,
                message=f"Daily loss {daily_pnl_pct:.2%} exceeds limit",
                current_value=abs(daily_pnl_pct),
                limit_value=self.settings.daily_loss_stop_pct,
            )

        if daily_pnl_pct < -self.settings.daily_loss_stop_pct * 0.8:
            return RiskCheck(
                name="daily_loss",
                status=RiskStatus.WARNING,
                message="Approaching daily loss limit",
                current_value=abs(daily_pnl_pct),
                limit_value=self.settings.daily_loss_stop_pct,
            )

        return RiskCheck(
            name="daily_loss",
            status=RiskStatus.OK,
            message="Daily loss OK",
            current_value=abs(daily_pnl_pct),
            limit_value=self.settings.daily_loss_stop_pct,
        )

    def _check_drawdown(self) -> RiskCheck:
        """Check maximum drawdown limit."""
        drawdown = self.current_drawdown

        if drawdown >= self.settings.max_drawdown_pct:
            self.trigger_emergency_stop("Maximum drawdown exceeded")
            return RiskCheck(
                name="drawdown",
                status=RiskStatus.BLOCKED,
                message=f"Drawdown {drawdown:.2%} exceeds limit",
                current_value=drawdown,
                limit_value=self.settings.max_drawdown_pct,
            )

        if drawdown >= self.settings.max_drawdown_pct * 0.8:
            return RiskCheck(
                name="drawdown",
                status=RiskStatus.WARNING,
                message="Approaching max drawdown",
                current_value=drawdown,
                limit_value=self.settings.max_drawdown_pct,
            )

        return RiskCheck(
            name="drawdown",
            status=RiskStatus.OK,
            message="Drawdown OK",
            current_value=drawdown,
            limit_value=self.settings.max_drawdown_pct,
        )

    def _check_model_agreement(self, agreement: float) -> RiskCheck:
        """Check model agreement threshold."""
        if agreement < self.settings.min_model_agreement:
            return RiskCheck(
                name="model_agreement",
                status=RiskStatus.BLOCKED,
                message=f"Model agreement {agreement:.2%} below threshold",
                current_value=agreement,
                limit_value=self.settings.min_model_agreement,
            )

        return RiskCheck(
            name="model_agreement",
            status=RiskStatus.OK,
            message="Model agreement OK",
            current_value=agreement,
            limit_value=self.settings.min_model_agreement,
        )

    def _check_liquidity(self, signal: TradeSignal) -> RiskCheck:
        """Check market liquidity."""
        spread = signal.bucket.spread

        # For weather markets, spreads can be wider due to lower liquidity
        # Block only if spread is extremely wide (>25%)
        if spread > 0.25:
            return RiskCheck(
                name="liquidity",
                status=RiskStatus.BLOCKED,
                message=f"Spread {spread:.2%} too wide",
                current_value=spread,
                limit_value=0.25,
            )

        # Warning if spread is concerning (>15%)
        if spread > 0.15:
            return RiskCheck(
                name="liquidity",
                status=RiskStatus.WARNING,
                message="Wide spread, may have slippage",
                current_value=spread,
                limit_value=0.25,
            )

        return RiskCheck(
            name="liquidity",
            status=RiskStatus.OK,
            message="Liquidity OK",
            current_value=spread,
            limit_value=0.25,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current risk status."""
        return {
            "current_bankroll": str(self._current_bankroll),
            "peak_bankroll": str(self._peak_bankroll),
            "current_drawdown": self.current_drawdown,
            "total_position_size": str(self.total_position_size),
            "n_positions": len(self._positions),
            "is_emergency_stopped": self._is_emergency_stopped,
            "today_pnl": self.today_pnl.pnl_pct,
            "today_trades": self.today_pnl.trades,
            "today_win_rate": self.today_pnl.win_rate,
        }

    def adjust_position_size(
        self,
        original_size: Decimal,
        checks: list[RiskCheck],
    ) -> Decimal:
        """
        Adjust position size based on risk checks.

        Reduces size if there are warnings.
        """
        adjusted = original_size

        # Reduce for warnings
        warnings = [c for c in checks if c.status == RiskStatus.WARNING]
        if warnings:
            # Reduce by 25% for each warning
            reduction = Decimal("0.75") ** len(warnings)
            adjusted = original_size * reduction

        # Ensure minimum viable size
        if adjusted < Decimal("1"):
            adjusted = Decimal("0")

        if adjusted != original_size:
            logger.info(
                "Adjusted position size",
                original=str(original_size),
                adjusted=str(adjusted),
                n_warnings=len(warnings),
            )

        return adjusted
