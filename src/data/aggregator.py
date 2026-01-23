"""Weather data aggregator combining multiple sources."""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import aiohttp
import numpy as np
from numpy.typing import NDArray

from src.config import CityConfig, get_settings
from src.data.ensemble import OpenMeteoClient, EnsembleForecast
from src.data.tomorrow import TomorrowClient, TomorrowForecast
from src.data.metar import MetarClient, MetarObservation
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedForecast:
    """
    Aggregated weather forecast from multiple sources.

    Combines ensemble forecasts, ML forecasts, and real-time observations
    to produce a unified probability distribution for temperature.
    """

    city: CityConfig
    target_date: date
    fetch_time: datetime

    # Primary ensemble forecast
    ensemble: EnsembleForecast | None = None

    # ML-enhanced forecast
    tomorrow: TomorrowForecast | None = None

    # Real-time observation (for nowcasting)
    metar: MetarObservation | None = None

    # Combined distribution
    combined_high_temps: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )

    # Model agreement metrics
    model_agreement: float = 0.0  # How well models agree (0-1)
    confidence: float = 0.0  # Overall forecast confidence (0-1)

    @property
    def n_sources(self) -> int:
        """Count of available data sources."""
        count = 0
        if self.ensemble is not None:
            count += 1
        if self.tomorrow is not None:
            count += 1
        if self.metar is not None:
            count += 1
        return count

    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for trading."""
        return self.ensemble is not None and self.n_sources >= 1

    def probability_in_range(self, low: float, high: float) -> float:
        """Calculate probability that high temp falls within range."""
        if len(self.combined_high_temps) == 0:
            return 0.0
        in_range = np.sum(
            (self.combined_high_temps >= low) & (self.combined_high_temps < high)
        )
        return float(in_range / len(self.combined_high_temps))

    def probability_above(self, threshold: float) -> float:
        """Calculate probability that high temp is at or above threshold."""
        if len(self.combined_high_temps) == 0:
            return 0.0
        return float(np.sum(self.combined_high_temps >= threshold) / len(self.combined_high_temps))

    def probability_below(self, threshold: float) -> float:
        """Calculate probability that high temp is below threshold."""
        if len(self.combined_high_temps) == 0:
            return 0.0
        return float(np.sum(self.combined_high_temps < threshold) / len(self.combined_high_temps))

    def get_distribution_stats(self) -> dict[str, float]:
        """Get distribution statistics."""
        if len(self.combined_high_temps) == 0:
            return {}
        return {
            "mean": float(np.mean(self.combined_high_temps)),
            "median": float(np.median(self.combined_high_temps)),
            "std": float(np.std(self.combined_high_temps)),
            "min": float(np.min(self.combined_high_temps)),
            "max": float(np.max(self.combined_high_temps)),
            "p10": float(np.percentile(self.combined_high_temps, 10)),
            "p25": float(np.percentile(self.combined_high_temps, 25)),
            "p75": float(np.percentile(self.combined_high_temps, 75)),
            "p90": float(np.percentile(self.combined_high_temps, 90)),
            "n_samples": len(self.combined_high_temps),
            "model_agreement": self.model_agreement,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        stats = self.get_distribution_stats()
        return {
            "city": self.city.name,
            "target_date": self.target_date.isoformat(),
            "fetch_time": self.fetch_time.isoformat(),
            "n_sources": self.n_sources,
            "has_ensemble": self.ensemble is not None,
            "has_tomorrow": self.tomorrow is not None,
            "has_metar": self.metar is not None,
            "model_agreement": self.model_agreement,
            "confidence": self.confidence,
            **stats,
        }


class WeatherAggregator:
    """
    Aggregates weather data from multiple sources.

    Combines:
    - Open-Meteo ensemble forecasts (200+ members)
    - Tomorrow.io ML forecasts (when available)
    - METAR real-time observations (for nowcasting)
    """

    def __init__(self):
        """Initialize the weather aggregator."""
        self.settings = get_settings()
        self._session: aiohttp.ClientSession | None = None
        self._open_meteo: OpenMeteoClient | None = None
        self._tomorrow: TomorrowClient | None = None
        self._metar: MetarClient | None = None

    async def __aenter__(self) -> "WeatherAggregator":
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        self._open_meteo = OpenMeteoClient(self._session)
        self._tomorrow = TomorrowClient(self._session)
        self._metar = MetarClient(self._session)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_forecast(
        self,
        city: CityConfig,
        target_date: date,
        include_metar: bool = True,
    ) -> AggregatedForecast:
        """
        Get aggregated forecast for a city and date.

        Args:
            city: City configuration
            target_date: The date to forecast
            include_metar: Whether to include real-time METAR data

        Returns:
            AggregatedForecast with combined probability distribution
        """
        if self._open_meteo is None:
            raise RuntimeError("WeatherAggregator must be used as context manager")

        result = AggregatedForecast(
            city=city,
            target_date=target_date,
            fetch_time=datetime.utcnow(),
        )

        # Fetch from all sources concurrently
        try:
            result.ensemble = await self._open_meteo.get_ensemble_forecast(
                city, target_date
            )
        except Exception as e:
            logger.error("Failed to fetch ensemble forecast", error=str(e))

        # Tomorrow.io (if available)
        if self._tomorrow and self._tomorrow.is_available:
            try:
                result.tomorrow = await self._tomorrow.get_forecast(city, target_date)
            except Exception as e:
                logger.warning("Failed to fetch Tomorrow.io forecast", error=str(e))

        # METAR (for nowcasting on target day or day before)
        if include_metar and self._metar:
            today = datetime.now().date()
            days_until = (target_date - today).days
            if days_until <= 1:  # Only fetch METAR for near-term dates
                try:
                    result.metar = await self._metar.get_observation(city)
                except Exception as e:
                    logger.warning("Failed to fetch METAR", error=str(e))

        # Combine distributions
        self._combine_forecasts(result)

        logger.info(
            "Aggregated weather forecast",
            city=city.name,
            target_date=str(target_date),
            n_sources=result.n_sources,
            model_agreement=result.model_agreement,
            confidence=result.confidence,
            stats=result.get_distribution_stats(),
        )

        return result

    def _combine_forecasts(self, result: AggregatedForecast) -> None:
        """Combine forecasts from multiple sources into unified distribution."""
        samples: list[NDArray[np.float64]] = []
        weights: list[float] = []

        # Ensemble forecast (primary source, highest weight)
        if result.ensemble is not None:
            samples.append(result.ensemble.high_temps)
            weights.append(1.0)

        # Tomorrow.io forecast (add synthetic samples around point forecast)
        if result.tomorrow is not None:
            # Create distribution around Tomorrow.io point forecast
            # Use tighter spread since it's ML-enhanced
            tomorrow_samples = np.random.normal(
                result.tomorrow.high_temp,
                2.0,  # Smaller std dev than ensemble
                50,
            )
            samples.append(tomorrow_samples)
            weights.append(0.5)  # Lower weight than ensemble

        # METAR nowcast (for same-day forecasts)
        if result.metar is not None and result.metar.is_fresh:
            # Use current temperature as strong signal for daily high
            # Assume current temp is within ~5 degrees of daily high
            current_temp = result.metar.get_temperature(result.city.unit)
            # The daily high is typically at or above current temp in afternoon
            metar_samples = np.random.normal(current_temp + 2, 1.5, 30)
            samples.append(metar_samples)
            weights.append(0.3)

        if not samples:
            result.combined_high_temps = np.array([])
            result.model_agreement = 0.0
            result.confidence = 0.0
            return

        # Weight and combine samples
        combined: list[float] = []
        total_weight = sum(weights)
        for sample_arr, weight in zip(samples, weights):
            # Number of samples proportional to weight
            n_samples = int(100 * weight / total_weight)
            if len(sample_arr) >= n_samples:
                # Random sample from the distribution
                indices = np.random.choice(len(sample_arr), n_samples, replace=False)
                combined.extend(sample_arr[indices].tolist())
            else:
                combined.extend(sample_arr.tolist())

        result.combined_high_temps = np.array(combined)

        # Calculate model agreement
        result.model_agreement = self._calculate_agreement(samples)

        # Calculate overall confidence
        result.confidence = self._calculate_confidence(result)

    def _calculate_agreement(
        self, samples: list[NDArray[np.float64]]
    ) -> float:
        """
        Calculate how well the models agree.

        Returns a value between 0 and 1, where 1 means perfect agreement.
        """
        if len(samples) < 2:
            return 1.0  # Only one source, perfect agreement with itself

        # Compare means of each source
        means = [float(np.mean(s)) for s in samples]
        stds = [float(np.std(s)) for s in samples]

        # Agreement is higher when means are close and stds are small
        mean_spread = max(means) - min(means)
        avg_std = np.mean(stds)

        # Normalize: if mean spread is within 2 degrees and std < 3, high agreement
        mean_agreement = max(0, 1 - mean_spread / 10)  # 10 degree range = 0 agreement
        std_agreement = max(0, 1 - avg_std / 5)  # 5 degree std = 0 agreement

        return float((mean_agreement + std_agreement) / 2)

    def _calculate_confidence(self, result: AggregatedForecast) -> float:
        """Calculate overall forecast confidence."""
        confidence = 0.0

        # Base confidence from number of sources
        confidence += 0.3 * min(result.n_sources / 3, 1.0)

        # Ensemble size contribution
        if result.ensemble:
            confidence += 0.3 * min(result.ensemble.n_members / 100, 1.0)

        # Model agreement contribution
        confidence += 0.2 * result.model_agreement

        # Narrow distribution = higher confidence
        if len(result.combined_high_temps) > 0:
            std = float(np.std(result.combined_high_temps))
            # Std of 2-3 degrees is very tight, 10+ is wide
            spread_confidence = max(0, 1 - std / 8)
            confidence += 0.2 * spread_confidence

        return min(confidence, 1.0)
