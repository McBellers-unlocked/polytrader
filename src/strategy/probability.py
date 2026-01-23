"""Fair value calculation using KDE probability distributions.

Computes fair probabilities for temperature buckets using:
- Kernel Density Estimation (KDE) from ensemble members
- METAR nowcasting constraints
- Model agreement scoring
- Edge detection and arbitrage opportunities
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Try importing scipy, fall back to simple histogram if unavailable
try:
    from scipy import stats
    from scipy import integrate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FairValueResult:
    """
    Result of fair value calculation for a temperature bucket.
    """
    # Bucket identification
    token_id: str
    outcome: str
    low_bound: float | None
    high_bound: float | None

    # Probability estimates
    fair_probability: float  # Our estimated probability (0-1)
    market_probability: float  # Market's implied probability (YES price)

    # Edge metrics
    edge: float  # (fair - market) / market
    edge_dollars: float  # Expected profit per dollar wagered
    expected_value: float  # EV of buying YES at market price

    # Confidence metrics
    model_agreement: float  # How well models agree (0-1)
    sample_size: int  # Number of ensemble members
    confidence: float  # Overall confidence in estimate

    # Distribution info
    kde_mean: float = 0.0
    kde_std: float = 0.0

    @property
    def has_positive_edge(self) -> bool:
        """Check if there's positive expected value."""
        return self.edge > 0.10  # 10% edge threshold

    @property
    def is_tradeable(self) -> bool:
        """Check if this meets trading criteria."""
        return (
            abs(self.edge) > 0.10 and
            self.model_agreement > 0.65 and
            self.confidence > 0.5
        )

    @property
    def signal(self) -> str:
        """Trading signal: BUY, SELL, or HOLD."""
        if not self.is_tradeable:
            return "HOLD"
        return "BUY" if self.edge > 0 else "SELL"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "outcome": self.outcome,
            "low_bound": self.low_bound,
            "high_bound": self.high_bound,
            "fair_probability": round(self.fair_probability, 4),
            "market_probability": round(self.market_probability, 4),
            "edge": round(self.edge, 4),
            "edge_dollars": round(self.edge_dollars, 4),
            "expected_value": round(self.expected_value, 4),
            "model_agreement": round(self.model_agreement, 4),
            "confidence": round(self.confidence, 4),
            "signal": self.signal,
            "is_tradeable": self.is_tradeable,
        }


@dataclass
class MarketFairValue:
    """
    Fair value analysis for an entire weather market.
    """
    condition_id: str
    city: str
    target_date: date
    analysis_time: datetime

    # Individual bucket results
    buckets: list[FairValueResult] = field(default_factory=list)

    # Market-level metrics
    total_fair_probability: float = 0.0  # Should be ~1.0
    total_market_probability: float = 0.0  # Sum of YES prices

    # Arbitrage opportunity
    arbitrage_gap: float = 0.0  # 1.0 - total_market_probability
    has_arbitrage: bool = False

    # Best opportunities
    best_buy: FairValueResult | None = None
    best_sell: FairValueResult | None = None

    # Distribution summary
    forecast_mean: float = 0.0
    forecast_std: float = 0.0
    model_agreement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "city": self.city,
            "target_date": self.target_date.isoformat(),
            "analysis_time": self.analysis_time.isoformat(),
            "total_fair_probability": round(self.total_fair_probability, 4),
            "total_market_probability": round(self.total_market_probability, 4),
            "arbitrage_gap": round(self.arbitrage_gap, 4),
            "has_arbitrage": self.has_arbitrage,
            "forecast_mean": round(self.forecast_mean, 2),
            "forecast_std": round(self.forecast_std, 2),
            "model_agreement": round(self.model_agreement, 4),
            "n_tradeable": sum(1 for b in self.buckets if b.is_tradeable),
            "best_buy_edge": round(self.best_buy.edge, 4) if self.best_buy else None,
            "best_sell_edge": round(self.best_sell.edge, 4) if self.best_sell else None,
            "buckets": [b.to_dict() for b in self.buckets],
        }


@dataclass
class ArbitrageOpportunity:
    """
    Rebalancing arbitrage opportunity.

    When sum of YES prices < $1.00, buying all YES tokens
    guarantees profit at resolution.
    """
    condition_id: str
    city: str
    target_date: date

    total_yes_price: float  # Sum of all YES prices
    gap: float  # 1.0 - total_yes_price
    profit_margin: float  # gap / total_yes_price

    # What to buy
    buckets_to_buy: list[tuple[str, str, float]]  # (token_id, outcome, price)

    # Sizing
    required_capital: Decimal = Decimal("0")
    guaranteed_profit: Decimal = Decimal("0")
    roi_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "city": self.city,
            "target_date": self.target_date.isoformat(),
            "total_yes_price": round(self.total_yes_price, 4),
            "gap": round(self.gap, 4),
            "profit_margin": round(self.profit_margin, 4),
            "required_capital": str(self.required_capital),
            "guaranteed_profit": str(self.guaranteed_profit),
            "roi_percent": round(self.roi_percent, 2),
            "n_buckets": len(self.buckets_to_buy),
        }


class FairValueCalculator:
    """
    Calculates fair value probabilities using KDE.

    Uses Gaussian Kernel Density Estimation to build a smooth
    probability distribution from ensemble forecast members,
    then integrates over bucket bounds to get fair probabilities.

    Usage:
        calc = FairValueCalculator()
        result = calc.calculate_fair_value(
            temperatures=forecast.temperatures,
            bucket_low=68.0,
            bucket_high=72.0,
            market_price=0.25,
            model_agreement=0.75,
        )
        print(f"Fair value: {result.fair_probability:.1%}, Edge: {result.edge:.1%}")
    """

    def __init__(self, bandwidth: str | float = "scott"):
        """
        Initialize the calculator.

        Args:
            bandwidth: KDE bandwidth selection method ('scott', 'silverman', or float)
        """
        self.bandwidth = bandwidth
        self.settings = get_settings()

    def calculate_fair_value(
        self,
        temperatures: NDArray[np.float64],
        bucket_low: float | None,
        bucket_high: float | None,
        market_price: float,
        model_agreement: float = 1.0,
        token_id: str = "",
        outcome: str = "",
        current_temp: float | None = None,  # METAR observation
    ) -> FairValueResult:
        """
        Calculate fair value for a single temperature bucket.

        Args:
            temperatures: Array of ensemble temperature forecasts
            bucket_low: Lower bound of bucket (inclusive), None for unbounded
            bucket_high: Upper bound of bucket (exclusive), None for unbounded
            market_price: Current YES price in market
            model_agreement: How well models agree (0-1)
            token_id: Token ID for the bucket
            outcome: Outcome string (e.g., "68-72°F")
            current_temp: Current observed temperature for nowcasting

        Returns:
            FairValueResult with probability and edge calculations
        """
        if len(temperatures) == 0:
            return self._empty_result(token_id, outcome, bucket_low, bucket_high, market_price)

        # Apply METAR nowcasting constraint if provided
        if current_temp is not None:
            temperatures = self._apply_metar_constraint(temperatures, current_temp)

        # Calculate probability using KDE or histogram
        if HAS_SCIPY and len(temperatures) >= 10:
            fair_prob, kde_mean, kde_std = self._kde_probability(
                temperatures, bucket_low, bucket_high
            )
        else:
            fair_prob = self._histogram_probability(temperatures, bucket_low, bucket_high)
            kde_mean = float(np.mean(temperatures))
            kde_std = float(np.std(temperatures))

        # Calculate edge
        if market_price > 0:
            edge = (fair_prob - market_price) / market_price
        else:
            edge = float('inf') if fair_prob > 0 else 0

        # Calculate expected value of buying YES at market price
        # EV = P(win) * payout - P(lose) * cost
        # Buying YES at price p: win pays (1-p), lose costs p
        ev = fair_prob * (1 - market_price) - (1 - fair_prob) * market_price

        # Edge in dollars (per $1 bet)
        edge_dollars = fair_prob - market_price

        # Confidence based on sample size and model agreement
        sample_confidence = min(len(temperatures) / 100, 1.0)
        confidence = model_agreement * sample_confidence

        return FairValueResult(
            token_id=token_id,
            outcome=outcome,
            low_bound=bucket_low,
            high_bound=bucket_high,
            fair_probability=fair_prob,
            market_probability=market_price,
            edge=edge,
            edge_dollars=edge_dollars,
            expected_value=ev,
            model_agreement=model_agreement,
            sample_size=len(temperatures),
            confidence=confidence,
            kde_mean=kde_mean,
            kde_std=kde_std,
        )

    def _kde_probability(
        self,
        temperatures: NDArray[np.float64],
        low: float | None,
        high: float | None,
    ) -> tuple[float, float, float]:
        """Calculate probability using Gaussian KDE."""
        try:
            # Fit KDE
            kde = stats.gaussian_kde(temperatures, bw_method=self.bandwidth)

            # Define integration bounds
            temp_min = float(np.min(temperatures)) - 20
            temp_max = float(np.max(temperatures)) + 20

            if low is None:
                low = temp_min
            if high is None:
                high = temp_max

            # Create wrapper function to ensure scalar output
            def kde_func(x: float) -> float:
                result = kde(x)
                if hasattr(result, '__len__'):
                    return float(result[0])
                return float(result)

            # Integrate KDE over bucket range
            probability, _ = integrate.quad(kde_func, low, high)

            # Clamp to valid range
            probability = max(0.0, min(1.0, probability))

            # Get KDE statistics
            kde_mean = float(np.mean(temperatures))
            kde_std = float(np.std(temperatures))

            return probability, kde_mean, kde_std

        except Exception as e:
            logger.warning(f"KDE failed, falling back to histogram: {e}")
            prob = self._histogram_probability(temperatures, low, high)
            return prob, float(np.mean(temperatures)), float(np.std(temperatures))

    def _histogram_probability(
        self,
        temperatures: NDArray[np.float64],
        low: float | None,
        high: float | None,
    ) -> float:
        """Calculate probability using simple histogram counting."""
        if low is None:
            low = float('-inf')
        if high is None:
            high = float('inf')

        in_range = np.sum((temperatures >= low) & (temperatures < high))
        return float(in_range / len(temperatures))

    def _apply_metar_constraint(
        self,
        temperatures: NDArray[np.float64],
        current_temp: float,
    ) -> NDArray[np.float64]:
        """
        Apply METAR nowcasting constraint.

        For "highest temperature" markets, the actual high can only be
        >= the current observed temperature. This truncates the distribution.

        Args:
            temperatures: Original ensemble temperatures
            current_temp: Current observed temperature

        Returns:
            Constrained temperature array
        """
        # Remove ensemble members below current temp
        # (they're already impossible)
        constrained = temperatures[temperatures >= current_temp]

        # If all members eliminated, keep original but shift up
        if len(constrained) < 5:
            shift = current_temp - np.min(temperatures) + 1
            constrained = temperatures + shift

        logger.debug(
            f"METAR constraint: current={current_temp:.1f}, "
            f"removed {len(temperatures) - len(constrained)} members"
        )

        return constrained

    def _empty_result(
        self,
        token_id: str,
        outcome: str,
        low: float | None,
        high: float | None,
        market_price: float,
    ) -> FairValueResult:
        """Return empty result when no data available."""
        return FairValueResult(
            token_id=token_id,
            outcome=outcome,
            low_bound=low,
            high_bound=high,
            fair_probability=0.0,
            market_probability=market_price,
            edge=0.0,
            edge_dollars=0.0,
            expected_value=0.0,
            model_agreement=0.0,
            sample_size=0,
            confidence=0.0,
        )

    def analyze_market(
        self,
        condition_id: str,
        city: str,
        target_date: date,
        temperatures: NDArray[np.float64],
        buckets: list[tuple[str, str, float | None, float | None, float]],  # (token_id, outcome, low, high, price)
        model_agreement: float = 1.0,
        current_temp: float | None = None,
    ) -> MarketFairValue:
        """
        Analyze all buckets in a market.

        Args:
            condition_id: Market condition ID
            city: City name
            target_date: Target forecast date
            temperatures: Ensemble temperature forecasts
            buckets: List of (token_id, outcome, low_bound, high_bound, market_price)
            model_agreement: Model agreement score
            current_temp: Current METAR observation

        Returns:
            MarketFairValue with all bucket analyses
        """
        result = MarketFairValue(
            condition_id=condition_id,
            city=city,
            target_date=target_date,
            analysis_time=datetime.utcnow(),
            forecast_mean=float(np.mean(temperatures)) if len(temperatures) > 0 else 0,
            forecast_std=float(np.std(temperatures)) if len(temperatures) > 0 else 0,
            model_agreement=model_agreement,
        )

        best_buy_edge = -float('inf')
        best_sell_edge = float('inf')

        for token_id, outcome, low, high, price in buckets:
            fv = self.calculate_fair_value(
                temperatures=temperatures,
                bucket_low=low,
                bucket_high=high,
                market_price=price,
                model_agreement=model_agreement,
                token_id=token_id,
                outcome=outcome,
                current_temp=current_temp,
            )
            result.buckets.append(fv)
            result.total_fair_probability += fv.fair_probability
            result.total_market_probability += fv.market_probability

            # Track best opportunities
            if fv.edge > best_buy_edge:
                best_buy_edge = fv.edge
                result.best_buy = fv
            if fv.edge < best_sell_edge:
                best_sell_edge = fv.edge
                result.best_sell = fv

        # Calculate arbitrage gap
        result.arbitrage_gap = 1.0 - result.total_market_probability
        result.has_arbitrage = abs(result.arbitrage_gap) > 0.02  # 2% threshold

        return result


class EdgeDetector:
    """
    Detects trading edge from fair value calculations.
    """

    def __init__(
        self,
        min_edge: float = 0.10,
        min_agreement: float = 0.65,
        min_confidence: float = 0.5,
    ):
        """
        Initialize edge detector.

        Args:
            min_edge: Minimum edge to consider trading (default 10%)
            min_agreement: Minimum model agreement required
            min_confidence: Minimum confidence required
        """
        self.min_edge = min_edge
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence

    def find_opportunities(
        self,
        market_analysis: MarketFairValue,
    ) -> list[FairValueResult]:
        """
        Find tradeable opportunities in a market.

        Args:
            market_analysis: MarketFairValue from calculator

        Returns:
            List of FairValueResult that meet criteria, sorted by edge
        """
        opportunities = []

        for bucket in market_analysis.buckets:
            if (
                abs(bucket.edge) >= self.min_edge and
                bucket.model_agreement >= self.min_agreement and
                bucket.confidence >= self.min_confidence
            ):
                opportunities.append(bucket)

        # Sort by absolute edge (best opportunities first)
        opportunities.sort(key=lambda x: abs(x.edge), reverse=True)

        return opportunities

    def calculate_kelly_fraction(
        self,
        fair_prob: float,
        market_price: float,
    ) -> float:
        """
        Calculate Kelly criterion fraction for position sizing.

        Kelly formula: f* = (bp - q) / b
        where b = odds, p = prob of winning, q = prob of losing

        Args:
            fair_prob: Our estimated probability
            market_price: Market YES price

        Returns:
            Optimal fraction of bankroll to bet (capped at 25%)
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # Odds for buying YES at price p
        b = (1 - market_price) / market_price
        p = fair_prob
        q = 1 - fair_prob

        kelly = (b * p - q) / b

        # Cap at 25% and ensure non-negative
        return max(0, min(kelly, 0.25))


class ArbitrageDetector:
    """
    Detects rebalancing arbitrage opportunities.

    When the sum of YES prices doesn't equal $1.00, there's
    a risk-free profit opportunity.
    """

    def __init__(self, min_gap: float = 0.02):
        """
        Initialize arbitrage detector.

        Args:
            min_gap: Minimum gap to consider (default 2%)
        """
        self.min_gap = min_gap

    def detect(
        self,
        condition_id: str,
        city: str,
        target_date: date,
        buckets: list[tuple[str, str, float]],  # (token_id, outcome, yes_price)
    ) -> ArbitrageOpportunity | None:
        """
        Detect arbitrage opportunity.

        Args:
            condition_id: Market condition ID
            city: City name
            target_date: Target date
            buckets: List of (token_id, outcome, yes_price)

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        total_yes = sum(price for _, _, price in buckets)
        gap = 1.0 - total_yes

        # Need gap > threshold for arbitrage
        if gap <= self.min_gap:
            return None

        profit_margin = gap / total_yes if total_yes > 0 else 0

        # Calculate capital required and profit
        required_capital = Decimal(str(total_yes))
        guaranteed_profit = Decimal(str(gap))
        roi = float(guaranteed_profit / required_capital) * 100 if required_capital > 0 else 0

        return ArbitrageOpportunity(
            condition_id=condition_id,
            city=city,
            target_date=target_date,
            total_yes_price=total_yes,
            gap=gap,
            profit_margin=profit_margin,
            buckets_to_buy=[(tid, out, price) for tid, out, price in buckets],
            required_capital=required_capital,
            guaranteed_profit=guaranteed_profit,
            roi_percent=roi,
        )

    def detect_reverse(
        self,
        condition_id: str,
        city: str,
        target_date: date,
        buckets: list[tuple[str, str, float, float]],  # (token_id, outcome, yes_price, no_price)
    ) -> ArbitrageOpportunity | None:
        """
        Detect reverse arbitrage (sum > $1.00).

        When YES prices sum to more than $1.00, buying all NO tokens
        is profitable.

        Args:
            condition_id: Market condition ID
            city: City name
            target_date: Target date
            buckets: List of (token_id, outcome, yes_price, no_price)

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        total_yes = sum(yes for _, _, yes, _ in buckets)
        excess = total_yes - 1.0

        if excess <= self.min_gap:
            return None

        total_no = sum(no for _, _, _, no in buckets)
        profit_margin = excess / total_no if total_no > 0 else 0

        required_capital = Decimal(str(total_no))
        guaranteed_profit = Decimal(str(excess))
        roi = float(guaranteed_profit / required_capital) * 100 if required_capital > 0 else 0

        return ArbitrageOpportunity(
            condition_id=condition_id,
            city=city,
            target_date=target_date,
            total_yes_price=total_yes,
            gap=-excess,  # Negative indicates reverse arb
            profit_margin=profit_margin,
            buckets_to_buy=[(tid, f"NO:{out}", no) for tid, out, _, no in buckets],
            required_capital=required_capital,
            guaranteed_profit=guaranteed_profit,
            roi_percent=roi,
        )


def demo_fair_value_calculation():
    """Demonstrate fair value calculation with mock data."""
    print(f"\n{'='*60}")
    print("Fair Value Calculator Demo")
    print(f"{'='*60}\n")

    # Create mock ensemble forecast (Dallas winter day)
    np.random.seed(42)
    temperatures = np.concatenate([
        np.random.normal(55, 3, 40),   # ICON
        np.random.normal(56, 4, 31),   # GFS
        np.random.normal(54, 2.5, 51), # ECMWF
    ])

    print(f"Ensemble Forecast:")
    print(f"  Members: {len(temperatures)}")
    print(f"  Mean: {np.mean(temperatures):.1f}°F")
    print(f"  Std: {np.std(temperatures):.1f}°F")
    print(f"  Range: [{np.min(temperatures):.1f}, {np.max(temperatures):.1f}]°F")

    # Mock market buckets
    buckets = [
        ("token_1", "<50°F", None, 50.0, 0.05),
        ("token_2", "50-55°F", 50.0, 55.0, 0.20),
        ("token_3", "55-60°F", 55.0, 60.0, 0.40),
        ("token_4", "60-65°F", 60.0, 65.0, 0.25),
        ("token_5", "≥65°F", 65.0, None, 0.08),
    ]

    print(f"\nMarket Buckets:")
    print(f"  Total YES prices: ${sum(b[4] for b in buckets):.3f}")

    # Calculate fair values
    calc = FairValueCalculator()
    result = calc.analyze_market(
        condition_id="demo_market",
        city="Dallas",
        target_date=date.today(),
        temperatures=temperatures,
        buckets=buckets,
        model_agreement=0.75,
    )

    print(f"\n{'='*60}")
    print("Fair Value Analysis")
    print(f"{'='*60}")

    print(f"\n{'Bucket':<12} {'Fair':<8} {'Market':<8} {'Edge':<10} {'Signal':<8}")
    print("-" * 50)

    for fv in result.buckets:
        print(
            f"{fv.outcome:<12} "
            f"{fv.fair_probability:>6.1%}  "
            f"{fv.market_probability:>6.1%}  "
            f"{fv.edge:>+8.1%}  "
            f"{fv.signal:<8}"
        )

    print("-" * 50)
    print(f"{'Total':<12} {result.total_fair_probability:>6.1%}  {result.total_market_probability:>6.1%}")

    # Check for arbitrage
    print(f"\nArbitrage Analysis:")
    print(f"  Gap: {result.arbitrage_gap:+.1%}")
    print(f"  Has Arbitrage: {result.has_arbitrage}")

    # Find best opportunities
    detector = EdgeDetector(min_edge=0.10)
    opps = detector.find_opportunities(result)

    print(f"\nTradeable Opportunities ({len(opps)}):")
    for opp in opps:
        kelly = detector.calculate_kelly_fraction(opp.fair_probability, opp.market_probability)
        print(f"  {opp.outcome}: edge={opp.edge:+.1%}, kelly={kelly:.1%}, EV=${opp.expected_value:.3f}")

    # Demo METAR constraint
    print(f"\n{'='*60}")
    print("METAR Nowcasting Demo")
    print(f"{'='*60}")

    current_temp = 52.0
    print(f"\nCurrent observed temperature: {current_temp}°F")
    print("(Temperature can only go higher from here)")

    result_metar = calc.analyze_market(
        condition_id="demo_market_metar",
        city="Dallas",
        target_date=date.today(),
        temperatures=temperatures,
        buckets=buckets,
        model_agreement=0.75,
        current_temp=current_temp,
    )

    print(f"\n{'Bucket':<12} {'Before':<8} {'After':<8} {'Change':<10}")
    print("-" * 40)

    for fv_before, fv_after in zip(result.buckets, result_metar.buckets):
        change = fv_after.fair_probability - fv_before.fair_probability
        print(
            f"{fv_before.outcome:<12} "
            f"{fv_before.fair_probability:>6.1%}  "
            f"{fv_after.fair_probability:>6.1%}  "
            f"{change:>+8.1%}"
        )

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    demo_fair_value_calculation()
