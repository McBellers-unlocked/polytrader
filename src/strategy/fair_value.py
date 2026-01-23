"""Fair value calculation from ensemble forecasts."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.aggregator import AggregatedForecast
from src.markets.scanner import WeatherMarket, TemperatureBucket
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BucketProbability:
    """
    Fair value probability for a temperature bucket.

    Calculated from ensemble weather forecast distribution.
    """

    bucket: TemperatureBucket
    fair_value: float  # Probability from ensemble (0-1)
    market_price: float  # Current market YES price (0-1)
    edge: float  # (fair_value - market_price) / market_price
    confidence: float  # Confidence in the fair value estimate

    @property
    def has_edge(self) -> bool:
        """Check if there's meaningful edge."""
        return abs(self.edge) > 0.05

    @property
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal (underpriced)."""
        return self.edge > 0

    @property
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal (overpriced)."""
        return self.edge < 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.bucket.outcome,
            "token_id": self.bucket.token_id,
            "low_bound": self.bucket.low_bound,
            "high_bound": self.bucket.high_bound,
            "fair_value": self.fair_value,
            "market_price": self.market_price,
            "edge": self.edge,
            "confidence": self.confidence,
        }


class FairValueCalculator:
    """
    Calculates fair value probabilities for temperature buckets.

    Uses the ensemble forecast distribution to estimate the probability
    that the actual temperature will fall within each bucket.
    """

    def __init__(self):
        """Initialize the calculator."""
        pass

    def calculate_fair_values(
        self,
        market: WeatherMarket,
        forecast: AggregatedForecast,
    ) -> list[BucketProbability]:
        """
        Calculate fair values for all buckets in a market.

        Args:
            market: The weather market with temperature buckets
            forecast: Aggregated forecast with probability distribution

        Returns:
            List of BucketProbability for each bucket
        """
        if not forecast.has_sufficient_data:
            logger.warning(
                "Insufficient forecast data",
                city=market.city.name if market.city else "unknown",
            )
            return []

        results: list[BucketProbability] = []

        for bucket in market.buckets:
            fair_value = self._calculate_bucket_probability(bucket, forecast)
            market_price = bucket.yes_price

            # Calculate edge
            if market_price > 0:
                edge = (fair_value - market_price) / market_price
            else:
                edge = float("inf") if fair_value > 0 else 0

            # Confidence is based on forecast confidence and sample size
            confidence = forecast.confidence * min(
                len(forecast.combined_high_temps) / 50, 1.0
            )

            prob = BucketProbability(
                bucket=bucket,
                fair_value=fair_value,
                market_price=market_price,
                edge=edge,
                confidence=confidence,
            )

            results.append(prob)

        # Log summary
        total_fair_value = sum(p.fair_value for p in results)
        total_market_price = sum(p.market_price for p in results)

        logger.info(
            "Calculated fair values",
            city=market.city.name if market.city else "unknown",
            n_buckets=len(results),
            total_fair_value=total_fair_value,
            total_market_price=total_market_price,
            forecast_mean=float(np.mean(forecast.combined_high_temps)),
            forecast_std=float(np.std(forecast.combined_high_temps)),
        )

        return results

    def _calculate_bucket_probability(
        self,
        bucket: TemperatureBucket,
        forecast: AggregatedForecast,
    ) -> float:
        """Calculate probability for a single bucket."""
        temps = forecast.combined_high_temps

        if len(temps) == 0:
            return 0.0

        # Count samples in bucket range
        low = bucket.low_bound if bucket.low_bound is not None else float("-inf")
        high = bucket.high_bound if bucket.high_bound is not None else float("inf")

        in_range = np.sum((temps >= low) & (temps < high))
        probability = float(in_range / len(temps))

        # Debug logging for unusual probabilities (helps diagnose data issues)
        if probability > 0.7 or probability < 0.01:
            temp_mean = float(np.mean(temps))
            temp_std = float(np.std(temps))
            temp_min = float(np.min(temps))
            temp_max = float(np.max(temps))
            logger.debug(
                "Bucket probability calculation",
                outcome=bucket.outcome,
                bucket_low=low,
                bucket_high=high,
                bucket_unit=bucket.unit,
                probability=f"{probability:.1%}",
                ensemble_mean=f"{temp_mean:.1f}",
                ensemble_std=f"{temp_std:.1f}",
                ensemble_range=f"{temp_min:.1f}-{temp_max:.1f}",
                n_samples=len(temps),
            )

        return probability

    def get_expected_value(
        self,
        bucket_probs: list[BucketProbability],
    ) -> float:
        """
        Calculate expected temperature from bucket probabilities.

        Useful for sanity checking the forecast.
        """
        total_prob = 0.0
        weighted_sum = 0.0

        for bp in bucket_probs:
            if bp.bucket.low_bound is not None and bp.bucket.high_bound is not None:
                midpoint = (bp.bucket.low_bound + bp.bucket.high_bound) / 2
                weighted_sum += midpoint * bp.fair_value
                total_prob += bp.fair_value

        if total_prob > 0:
            return weighted_sum / total_prob
        return 0.0

    def calibrate_probabilities(
        self,
        bucket_probs: list[BucketProbability],
    ) -> list[BucketProbability]:
        """
        Calibrate probabilities to sum to 1.0.

        Sometimes due to binning or rounding, probabilities don't
        sum exactly to 1.0. This normalizes them.
        """
        total = sum(bp.fair_value for bp in bucket_probs)

        if total == 0 or abs(total - 1.0) < 0.001:
            return bucket_probs

        # Normalize
        for bp in bucket_probs:
            bp.fair_value = bp.fair_value / total

            # Recalculate edge
            if bp.market_price > 0:
                bp.edge = (bp.fair_value - bp.market_price) / bp.market_price

        return bucket_probs
