"""Arbitrage detection for rebalancing opportunities."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.markets.scanner import WeatherMarket, TemperatureBucket
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    """
    An arbitrage opportunity from bucket mispricing.

    When the sum of all YES prices is less than $1.00, buying
    all YES tokens guarantees profit at resolution.
    """

    market: WeatherMarket
    total_yes_price: float  # Sum of all YES prices
    profit_margin: float  # (1.0 - total_yes_price) / total_yes_price
    required_capital: Decimal  # Capital needed to buy all YES
    guaranteed_profit: Decimal  # Profit at resolution
    buckets_to_buy: list[tuple[TemperatureBucket, float]]  # (bucket, price)

    @property
    def roi(self) -> float:
        """Return on investment percentage."""
        if self.required_capital > 0:
            return float(self.guaranteed_profit / self.required_capital) * 100
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.market.condition_id,
            "total_yes_price": self.total_yes_price,
            "profit_margin": self.profit_margin,
            "required_capital": str(self.required_capital),
            "guaranteed_profit": str(self.guaranteed_profit),
            "roi_pct": self.roi,
            "n_buckets": len(self.buckets_to_buy),
        }


class ArbitrageDetector:
    """
    Detects arbitrage opportunities in weather markets.

    Looks for markets where the sum of YES prices doesn't equal $1.00,
    which creates risk-free profit opportunities.
    """

    # Minimum profit margin to consider (accounts for fees)
    MIN_PROFIT_MARGIN = 0.02  # 2%

    # Maximum capital to deploy in single arb
    MAX_ARB_CAPITAL = Decimal("100")

    def __init__(self):
        """Initialize the arbitrage detector."""
        pass

    def detect_arbitrage(
        self,
        market: WeatherMarket,
    ) -> ArbitrageOpportunity | None:
        """
        Detect arbitrage opportunity in a market.

        The key insight: if sum of all YES prices < $1.00,
        buying all YES tokens guarantees that exactly one
        will pay out $1.00 at resolution.

        Args:
            market: The weather market to analyze

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        if not market.buckets:
            return None

        # Calculate sum of YES prices
        total_yes = sum(b.yes_price for b in market.buckets)

        # Calculate the gap
        gap = 1.0 - total_yes

        if gap <= self.MIN_PROFIT_MARGIN:
            return None

        # This is an arbitrage opportunity
        profit_margin = gap / total_yes if total_yes > 0 else 0

        # Calculate capital needed (buy 1 share of each YES)
        required_capital = Decimal(str(total_yes))

        # Guaranteed profit is the gap
        guaranteed_profit = Decimal(str(gap))

        # Collect buckets to buy
        buckets_to_buy = [(b, b.yes_price) for b in market.buckets]

        arb = ArbitrageOpportunity(
            market=market,
            total_yes_price=total_yes,
            profit_margin=profit_margin,
            required_capital=required_capital,
            guaranteed_profit=guaranteed_profit,
            buckets_to_buy=buckets_to_buy,
        )

        logger.info(
            "Detected arbitrage opportunity",
            condition_id=market.condition_id,
            total_yes_price=total_yes,
            profit_margin=profit_margin,
            roi_pct=arb.roi,
        )

        return arb

    def detect_reverse_arbitrage(
        self,
        market: WeatherMarket,
    ) -> ArbitrageOpportunity | None:
        """
        Detect reverse arbitrage (sum > $1.00).

        If sum of YES prices > $1.00, selling all YES tokens
        (or buying all NO tokens) guarantees profit.

        Args:
            market: The weather market to analyze

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        if not market.buckets:
            return None

        total_yes = sum(b.yes_price for b in market.buckets)

        # Looking for total > 1.0
        excess = total_yes - 1.0

        if excess <= self.MIN_PROFIT_MARGIN:
            return None

        profit_margin = excess / (total_yes - excess)  # Profit relative to cost

        # To profit: sell YES across all buckets
        # Or equivalently: buy NO across all buckets
        total_no = sum(b.no_price for b in market.buckets)
        required_capital = Decimal(str(total_no))
        guaranteed_profit = Decimal(str(excess))

        buckets_to_buy = [(b, b.no_price) for b in market.buckets]

        arb = ArbitrageOpportunity(
            market=market,
            total_yes_price=total_yes,
            profit_margin=profit_margin,
            required_capital=required_capital,
            guaranteed_profit=guaranteed_profit,
            buckets_to_buy=buckets_to_buy,
        )

        logger.info(
            "Detected reverse arbitrage opportunity",
            condition_id=market.condition_id,
            total_yes_price=total_yes,
            profit_margin=profit_margin,
            roi_pct=arb.roi,
        )

        return arb

    def find_all_arbitrage(
        self,
        markets: list[WeatherMarket],
    ) -> list[ArbitrageOpportunity]:
        """
        Find all arbitrage opportunities across markets.

        Args:
            markets: List of markets to analyze

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities: list[ArbitrageOpportunity] = []

        for market in markets:
            # Check for standard arbitrage (sum < 1)
            arb = self.detect_arbitrage(market)
            if arb:
                opportunities.append(arb)

            # Check for reverse arbitrage (sum > 1)
            reverse_arb = self.detect_reverse_arbitrage(market)
            if reverse_arb:
                opportunities.append(reverse_arb)

        # Sort by ROI
        opportunities.sort(key=lambda a: a.roi, reverse=True)

        if opportunities:
            logger.info(
                "Found arbitrage opportunities",
                n_opportunities=len(opportunities),
                best_roi=opportunities[0].roi if opportunities else 0,
            )

        return opportunities

    def calculate_optimal_size(
        self,
        arb: ArbitrageOpportunity,
        available_capital: Decimal,
        max_position_pct: float = 0.02,
    ) -> Decimal:
        """
        Calculate optimal position size for arbitrage.

        Args:
            arb: The arbitrage opportunity
            available_capital: Total available capital
            max_position_pct: Maximum position as percentage of capital

        Returns:
            Optimal position size
        """
        # Max based on position limit
        max_from_limit = available_capital * Decimal(str(max_position_pct))

        # Max based on arb capital requirement
        max_from_arb = self.MAX_ARB_CAPITAL

        # Take the minimum
        return min(max_from_limit, max_from_arb, arb.required_capital)
