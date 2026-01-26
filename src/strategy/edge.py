"""Edge detection and trade signal generation."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from src.config import get_settings
from src.markets.scanner import WeatherMarket, TemperatureBucket
from src.strategy.fair_value import BucketProbability
from src.logging import get_logger

logger = get_logger(__name__)


class SignalType(str, Enum):
    """Type of trade signal."""

    BUY_YES = "BUY_YES"  # Buy YES tokens (bet on this outcome)
    SELL_YES = "SELL_YES"  # Sell YES tokens (bet against this outcome)
    BUY_NO = "BUY_NO"  # Buy NO tokens
    SELL_NO = "SELL_NO"  # Sell NO tokens


@dataclass
class TradeSignal:
    """
    A trade signal generated from edge detection.

    Contains all information needed to execute the trade.
    """

    market: WeatherMarket
    bucket: TemperatureBucket
    signal_type: SignalType
    edge: float  # Edge percentage
    fair_value: float
    market_price: float
    confidence: float
    suggested_size: Decimal  # Suggested position size
    kelly_fraction: float  # Kelly criterion fraction
    model_agreement: float

    @property
    def expected_value(self) -> float:
        """Calculate expected value of the trade."""
        # EV = (probability of winning * payout) - (probability of losing * stake)
        # For YES at price p with fair value fv:
        # Win: fv * (1 - p) / p
        # Lose: (1 - fv) * 1
        if self.signal_type == SignalType.BUY_YES:
            return self.fair_value * (1 - self.market_price) - (
                1 - self.fair_value
            ) * self.market_price
        elif self.signal_type == SignalType.SELL_YES:
            return (1 - self.fair_value) * self.market_price - self.fair_value * (
                1 - self.market_price
            )
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.market.condition_id,
            "token_id": self.bucket.token_id,
            "outcome": self.bucket.outcome,
            "signal_type": self.signal_type.value,
            "edge": self.edge,
            "fair_value": self.fair_value,
            "market_price": self.market_price,
            "confidence": self.confidence,
            "suggested_size": str(self.suggested_size),
            "kelly_fraction": self.kelly_fraction,
            "model_agreement": self.model_agreement,
            "expected_value": self.expected_value,
        }


class EdgeDetector:
    """
    Detects trading edge and generates trade signals.

    Uses fair value calculations to identify mispriced markets
    and generates signals when edge exceeds threshold.
    """

    def __init__(self):
        """Initialize the edge detector."""
        self.settings = get_settings()
        # Track existing positions to avoid conflicts
        self._existing_positions: dict[str, str] = {}  # token_id -> side

    def set_existing_positions(self, positions: dict[str, str]) -> None:
        """
        Set existing positions to avoid conflicts.

        Args:
            positions: Dict mapping token_id to side ("YES" or "NO")
        """
        self._existing_positions = positions

    def detect_signals(
        self,
        market: WeatherMarket,
        bucket_probs: list[BucketProbability],
        model_agreement: float,
        bankroll: Decimal,
    ) -> list[TradeSignal]:
        """
        Detect trade signals from bucket probabilities.

        Args:
            market: The weather market
            bucket_probs: Fair value probabilities for each bucket
            model_agreement: How well weather models agree (0-1)
            bankroll: Current bankroll for position sizing

        Returns:
            List of TradeSignal objects for buckets with edge
        """
        signals: list[TradeSignal] = []

        for bp in bucket_probs:
            signal = self._evaluate_bucket(market, bp, model_agreement, bankroll)
            if signal:
                # Check for conflicting positions
                if self._has_conflict(signal):
                    logger.warning(
                        "Skipping signal due to conflicting position",
                        outcome=bp.bucket.outcome,
                        signal_type=signal.signal_type.value,
                    )
                    continue
                signals.append(signal)

        # Sort by edge * confidence (highest first)
        signals.sort(key=lambda s: abs(s.edge) * s.confidence, reverse=True)

        # Limit positions per market
        signals = signals[: self.settings.max_positions_per_market]

        if signals:
            logger.info(
                "Detected trade signals",
                market=market.condition_id,
                n_signals=len(signals),
                best_edge=signals[0].edge if signals else 0,
            )

        return signals

    def _has_conflict(self, signal: TradeSignal) -> bool:
        """Check if signal conflicts with existing position."""
        token_id = signal.bucket.token_id
        if token_id not in self._existing_positions:
            return False

        existing_side = self._existing_positions[token_id]
        new_side = "YES" if signal.signal_type == SignalType.BUY_YES else "NO"

        # Conflict if trying to take opposite side
        return existing_side != new_side

    def _evaluate_bucket(
        self,
        market: WeatherMarket,
        bp: BucketProbability,
        model_agreement: float,
        bankroll: Decimal,
    ) -> TradeSignal | None:
        """Evaluate a single bucket for trading edge."""
        # Check minimum edge threshold (STRICT: must be positive edge)
        if bp.edge < self.settings.min_edge_threshold:
            logger.debug(
                "Skipping signal due to insufficient edge",
                outcome=bp.bucket.outcome,
                edge=bp.edge,
                threshold=self.settings.min_edge_threshold,
            )
            return None

        # Check model agreement
        if model_agreement < self.settings.min_model_agreement:
            logger.debug(
                "Skipping signal due to low model agreement",
                outcome=bp.bucket.outcome,
                model_agreement=model_agreement,
            )
            return None

        # Check confidence (increased threshold)
        if bp.confidence < self.settings.min_confidence:
            logger.debug(
                "Skipping signal due to low confidence",
                outcome=bp.bucket.outcome,
                confidence=bp.confidence,
            )
            return None

        # Check liquidity (bid-ask spread)
        spread = bp.bucket.spread if hasattr(bp.bucket, 'spread') else 0
        if spread > self.settings.max_spread_pct:
            logger.debug(
                "Skipping signal due to wide spread",
                outcome=bp.bucket.outcome,
                spread=spread,
            )
            return None

        # Check minimum liquidity
        volume = bp.bucket.volume_24h if hasattr(bp.bucket, 'volume_24h') else 0
        if volume < self.settings.min_liquidity_usd:
            logger.debug(
                "Skipping signal due to low liquidity",
                outcome=bp.bucket.outcome,
                volume=volume,
            )
            return None

        # Only generate BUY_YES signals for positive edge
        # (we identified that NO positions are underperforming)
        if bp.edge > 0:
            signal_type = SignalType.BUY_YES
        else:
            # Skip negative edge signals entirely
            return None

        # Calculate Kelly fraction
        kelly = self._calculate_kelly(bp.fair_value, bp.market_price, signal_type)

        # Apply quarter-Kelly for extra safety (reduced from half)
        quarter_kelly = kelly / 4

        # Calculate suggested position size
        max_position = float(bankroll) * self.settings.max_position_pct
        kelly_position = float(bankroll) * quarter_kelly
        suggested_size = Decimal(str(min(kelly_position, max_position)))

        # Ensure minimum viable trade size ($5 minimum)
        if suggested_size < Decimal("5"):
            return None

        signal = TradeSignal(
            market=market,
            bucket=bp.bucket,
            signal_type=signal_type,
            edge=bp.edge,
            fair_value=bp.fair_value,
            market_price=bp.market_price,
            confidence=bp.confidence,
            suggested_size=suggested_size,
            kelly_fraction=quarter_kelly,
            model_agreement=model_agreement,
        )

        logger.info(
            "Generated trade signal",
            outcome=bp.bucket.outcome,
            signal_type=signal_type.value,
            edge=f"{bp.edge:.1%}",
            fair_value=f"{bp.fair_value:.1%}",
            market_price=f"{bp.market_price:.1%}",
            suggested_size=str(suggested_size),
        )

        return signal

    def _calculate_kelly(
        self,
        fair_value: float,
        market_price: float,
        signal_type: SignalType,
    ) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly formula: f* = (bp - q) / b
        where:
        - b = odds received (net odds)
        - p = probability of winning
        - q = probability of losing = 1 - p
        """
        if signal_type == SignalType.BUY_YES:
            # Buying YES at price p, win (1-p) if correct
            p = fair_value
            b = (1 - market_price) / market_price if market_price > 0 else 0
        else:
            # Selling YES (buying NO) at price (1-p)
            p = 1 - fair_value
            b = market_price / (1 - market_price) if market_price < 1 else 0

        if b <= 0:
            return 0.0

        q = 1 - p
        kelly = (b * p - q) / b

        # Cap Kelly at reasonable levels
        return max(0, min(kelly, 0.25))

    def filter_signals(
        self,
        signals: list[TradeSignal],
        max_signals: int = 3,
    ) -> list[TradeSignal]:
        """
        Filter signals to top opportunities.

        Args:
            signals: List of trade signals
            max_signals: Maximum signals to return

        Returns:
            Filtered list of best signals
        """
        if not signals:
            return []

        # Filter by minimum expected value
        signals = [s for s in signals if s.expected_value > 0]

        # Filter by minimum edge (15%)
        signals = [s for s in signals if s.edge >= 0.15]

        # Filter by minimum confidence (50%)
        signals = [s for s in signals if s.confidence >= 0.50]

        # Prefer cheap YES bets (asymmetric payoff)
        # Sort by: edge * confidence * (1 / market_price) to favor cheap bets
        def score(s: TradeSignal) -> float:
            price_factor = 1 / max(s.market_price, 0.01)  # Favor cheap prices
            return s.edge * s.confidence * min(price_factor, 10)  # Cap at 10x

        signals.sort(key=score, reverse=True)

        return signals[:max_signals]
