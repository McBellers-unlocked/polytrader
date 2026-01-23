"""METAR nowcasting for real-time probability constraint.

During trading hours (10am-6pm local time), METAR observations provide
crucial information: the final daily high temperature MUST be >= the
highest temperature observed so far today.

This narrows uncertainty significantly by afternoon, creating trading
opportunities when markets haven't fully priced in the constraint.

Usage:
    nowcaster = MetarNowcaster()
    await nowcaster.start()

    # In trading loop:
    constraint = nowcaster.get_constraint(city)
    if constraint.is_active:
        fair_value = calc.calculate_fair_value(
            temperatures=forecast.temperatures,
            bucket_low=55.0,
            bucket_high=60.0,
            market_price=0.25,
            current_temp=constraint.max_temp_observed,
        )
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from src.config import CityConfig, CITIES, get_settings
from src.data.metar import MetarClient, MetarObservation
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemperatureConstraint:
    """
    Nowcasting constraint for a city on a specific date.

    When active, provides a lower bound for the daily high temperature
    based on METAR observations.
    """
    city: str
    city_key: str
    target_date: date
    unit: str  # "F" or "C"

    # Observations
    max_temp_observed: float | None = None
    last_temp_observed: float | None = None
    last_observation_time: datetime | None = None
    observation_count: int = 0

    # Constraint status
    is_active: bool = False
    is_trading_hours: bool = False
    hours_until_close: float = 0.0

    # History of observations
    observations: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Confidence in the constraint (0-1)."""
        if not self.is_active or self.max_temp_observed is None:
            return 0.0

        # More observations = higher confidence
        obs_confidence = min(self.observation_count / 12, 1.0)  # 12 obs = full confidence

        # Later in day = higher confidence
        if self.hours_until_close < 2:
            time_confidence = 0.95  # Very high confidence near end
        elif self.hours_until_close < 4:
            time_confidence = 0.85
        elif self.hours_until_close < 6:
            time_confidence = 0.70
        else:
            time_confidence = 0.50

        return obs_confidence * time_confidence

    @property
    def uncertainty_reduction(self) -> float:
        """
        Estimated reduction in temperature uncertainty.

        As we observe max temp, uncertainty narrows. By afternoon,
        we often know the high within 1-2°F.
        """
        if not self.is_active:
            return 0.0

        # Base uncertainty from forecast is typically 4-5°F std
        # Afternoon observation narrows this significantly
        if self.hours_until_close < 2:
            return 0.90  # Know within ~1°F
        elif self.hours_until_close < 4:
            return 0.75  # Know within ~2°F
        elif self.hours_until_close < 6:
            return 0.50  # Know within ~3°F
        else:
            return 0.25  # Still useful constraint

    def to_dict(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "target_date": self.target_date.isoformat(),
            "unit": self.unit,
            "max_temp_observed": self.max_temp_observed,
            "last_temp_observed": self.last_temp_observed,
            "last_observation_time": (
                self.last_observation_time.isoformat()
                if self.last_observation_time else None
            ),
            "observation_count": self.observation_count,
            "is_active": self.is_active,
            "is_trading_hours": self.is_trading_hours,
            "hours_until_close": round(self.hours_until_close, 2),
            "confidence": round(self.confidence, 3),
            "uncertainty_reduction": round(self.uncertainty_reduction, 3),
        }


class MetarNowcaster:
    """
    Real-time METAR observation tracker for nowcasting.

    Fetches METAR observations every 15 minutes during trading hours
    (10am-6pm local time) and maintains running max temperature
    for probability constraint.

    The afternoon is the prime trading window because:
    - We often know within 1-2°F what the high will be
    - Markets haven't always fully updated
    - Strong edge when our constraint eliminates impossible outcomes
    """

    # Trading hours (local time for each city)
    TRADING_START = time(10, 0)  # 10:00 AM
    TRADING_END = time(18, 0)    # 6:00 PM

    # Update interval
    UPDATE_INTERVAL_SECONDS = 15 * 60  # 15 minutes

    def __init__(self, metar_client: MetarClient | None = None):
        """Initialize the nowcaster."""
        self.metar_client = metar_client or MetarClient()
        self._constraints: dict[str, TemperatureConstraint] = {}  # city_key -> constraint
        self._running = False
        self._update_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the nowcaster background task."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("METAR nowcaster started")

    async def stop(self) -> None:
        """Stop the nowcaster."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        await self.metar_client.close()
        logger.info("METAR nowcaster stopped")

    async def _update_loop(self) -> None:
        """Background loop to fetch METAR observations."""
        while self._running:
            try:
                await self._update_all_cities()

                # Wait for next update
                await asyncio.sleep(self.UPDATE_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Nowcaster update error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _update_all_cities(self) -> None:
        """Update observations for all tracked cities."""
        today = date.today()

        for city_key, city in CITIES.items():
            try:
                # Check if within trading hours for this city
                if not self._is_trading_hours(city):
                    continue

                # Fetch METAR
                observation = await self.metar_client.get_observation(city)
                if observation:
                    self._record_observation(city_key, city, today, observation)

            except Exception as e:
                logger.warning(f"Failed to update {city_key}: {e}")

    def _is_trading_hours(self, city: CityConfig) -> bool:
        """Check if current time is within trading hours for this city."""
        try:
            tz = ZoneInfo(city.timezone)
            local_now = datetime.now(tz)
            local_time = local_now.time()

            return self.TRADING_START <= local_time <= self.TRADING_END
        except Exception:
            # Default to UTC check if timezone fails
            utc_now = datetime.utcnow()
            return 10 <= utc_now.hour <= 18

    def _get_hours_until_close(self, city: CityConfig) -> float:
        """Get hours remaining until trading close for this city."""
        try:
            tz = ZoneInfo(city.timezone)
            local_now = datetime.now(tz)

            # Create trading end datetime for today
            trading_end = datetime.combine(local_now.date(), self.TRADING_END)
            trading_end = trading_end.replace(tzinfo=tz)

            if local_now >= trading_end:
                return 0.0

            delta = trading_end - local_now
            return delta.total_seconds() / 3600

        except Exception:
            return 8.0  # Default to 8 hours if timezone fails

    def _record_observation(
        self,
        city_key: str,
        city: CityConfig,
        target_date: date,
        observation: MetarObservation,
    ) -> None:
        """Record a METAR observation."""
        # Get or create constraint
        constraint = self._get_or_create_constraint(city_key, city, target_date)

        # Get temperature in city's unit
        temp = observation.get_temperature(city.unit)

        # Update constraint
        constraint.last_temp_observed = temp
        constraint.last_observation_time = observation.observation_time
        constraint.observation_count += 1
        constraint.observations.append((observation.observation_time, temp))

        # Update max
        if constraint.max_temp_observed is None or temp > constraint.max_temp_observed:
            constraint.max_temp_observed = temp
            logger.info(
                "New max temperature observed",
                city=city.name,
                max_temp=temp,
                unit=city.unit,
                observation_count=constraint.observation_count,
            )

        # Update trading status
        constraint.is_trading_hours = self._is_trading_hours(city)
        constraint.hours_until_close = self._get_hours_until_close(city)
        constraint.is_active = (
            constraint.is_trading_hours and
            constraint.max_temp_observed is not None
        )

    def _get_or_create_constraint(
        self,
        city_key: str,
        city: CityConfig,
        target_date: date,
    ) -> TemperatureConstraint:
        """Get or create a constraint for a city."""
        # Key includes date to reset daily
        key = f"{city_key}:{target_date.isoformat()}"

        if key not in self._constraints:
            self._constraints[key] = TemperatureConstraint(
                city=city.name,
                city_key=city_key,
                target_date=target_date,
                unit=city.unit,
            )

        return self._constraints[key]

    def get_constraint(
        self,
        city: CityConfig | str,
        target_date: date | None = None,
    ) -> TemperatureConstraint | None:
        """
        Get the current constraint for a city.

        Args:
            city: CityConfig or city key
            target_date: Target date (default: today)

        Returns:
            TemperatureConstraint or None if not available
        """
        if target_date is None:
            target_date = date.today()

        if isinstance(city, CityConfig):
            city_key = next(
                (k for k, v in CITIES.items() if v.name == city.name),
                None
            )
            if city_key is None:
                return None
        else:
            city_key = city

        key = f"{city_key}:{target_date.isoformat()}"
        return self._constraints.get(key)

    async def fetch_constraint(
        self,
        city: CityConfig,
        target_date: date | None = None,
    ) -> TemperatureConstraint | None:
        """
        Fetch and return constraint, updating if needed.

        This is the main method to call from the trading loop.
        """
        if target_date is None:
            target_date = date.today()

        # Only relevant for today's market
        if target_date != date.today():
            return None

        # Check if within trading hours
        if not self._is_trading_hours(city):
            return None

        # Get city key
        city_key = next(
            (k for k, v in CITIES.items() if v.name == city.name),
            None
        )
        if city_key is None:
            return None

        # Fetch fresh observation if needed
        constraint = self.get_constraint(city_key, target_date)

        if constraint is None or constraint.last_observation_time is None:
            # No observation yet, fetch one
            observation = await self.metar_client.get_observation(city)
            if observation:
                self._record_observation(city_key, city, target_date, observation)
                constraint = self.get_constraint(city_key, target_date)

        elif constraint.last_observation_time:
            # Check if observation is stale (> 20 minutes old)
            age = datetime.utcnow() - constraint.last_observation_time.replace(tzinfo=None)
            if age > timedelta(minutes=20):
                observation = await self.metar_client.get_observation(city)
                if observation:
                    self._record_observation(city_key, city, target_date, observation)
                    constraint = self.get_constraint(city_key, target_date)

        return constraint

    def apply_constraint(
        self,
        temperatures: NDArray[np.float64],
        constraint: TemperatureConstraint,
    ) -> NDArray[np.float64]:
        """
        Apply METAR constraint to temperature distribution.

        Removes ensemble members below the observed max and optionally
        shifts the distribution to account for the constraint.

        Args:
            temperatures: Ensemble temperature forecasts
            constraint: Active temperature constraint

        Returns:
            Constrained temperature array
        """
        if not constraint.is_active or constraint.max_temp_observed is None:
            return temperatures

        min_temp = constraint.max_temp_observed

        # Remove members below observed max
        constrained = temperatures[temperatures >= min_temp]

        # If too many members eliminated, shift distribution instead
        if len(constrained) < max(10, len(temperatures) * 0.2):
            # Shift entire distribution up
            shift = min_temp - np.min(temperatures) + 0.5
            constrained = temperatures + shift

            logger.debug(
                "Shifted distribution for METAR constraint",
                city=constraint.city,
                shift=shift,
                min_temp=min_temp,
            )
        else:
            logger.debug(
                "Applied METAR constraint",
                city=constraint.city,
                min_temp=min_temp,
                removed_count=len(temperatures) - len(constrained),
                remaining_count=len(constrained),
            )

        return constrained

    def get_all_constraints(self) -> dict[str, TemperatureConstraint]:
        """Get all active constraints."""
        return {
            k: v for k, v in self._constraints.items()
            if v.is_active
        }

    def get_status(self) -> dict[str, Any]:
        """Get nowcaster status."""
        active = [c for c in self._constraints.values() if c.is_active]
        return {
            "running": self._running,
            "total_constraints": len(self._constraints),
            "active_constraints": len(active),
            "cities_tracked": list(set(c.city for c in self._constraints.values())),
            "constraints": [c.to_dict() for c in active],
        }


def create_mock_constraint(
    city: CityConfig,
    max_temp: float | None = None,
    hours_until_close: float = 4.0,
) -> TemperatureConstraint:
    """
    Create a mock constraint for testing.

    Args:
        city: City configuration
        max_temp: Maximum temperature observed (default: simulated)
        hours_until_close: Hours until trading close
    """
    today = date.today()

    # Simulate realistic afternoon observation
    if max_temp is None:
        # Use typical values based on city
        if city.unit == "F":
            max_temp = 55.0 + np.random.uniform(-5, 5)
        else:
            max_temp = 13.0 + np.random.uniform(-3, 3)

    # Calculate observation count (4 per hour during 10am-6pm window)
    # hours_until_close of 4 means we're at 2pm = 4 hours of observations
    hours_elapsed = max(0, 8 - hours_until_close)  # 8 hour trading window
    observation_count = max(1, int(hours_elapsed * 4))  # ~4 obs per hour

    constraint = TemperatureConstraint(
        city=city.name,
        city_key=next((k for k, v in CITIES.items() if v.name == city.name), ""),
        target_date=today,
        unit=city.unit,
        max_temp_observed=max_temp,
        last_temp_observed=max_temp,
        last_observation_time=datetime.utcnow() - timedelta(minutes=10),
        observation_count=observation_count,
        is_active=True,
        is_trading_hours=True,
        hours_until_close=min(hours_until_close, 8.0),  # Cap at 8h trading window
    )

    return constraint


async def demo_nowcasting():
    """Demonstrate the METAR nowcasting system."""
    print(f"\n{'='*60}")
    print("METAR Nowcasting Demo")
    print(f"{'='*60}\n")

    # Create mock constraints for afternoon trading
    dallas = CITIES["dallas"]
    nyc = CITIES["nyc"]

    # Simulate 2pm observation (4 hours until close)
    dallas_constraint = create_mock_constraint(
        city=dallas,
        max_temp=58.0,  # Observed max so far
        hours_until_close=4.0,
    )

    # Simulate 4pm observation (2 hours until close)
    nyc_constraint = create_mock_constraint(
        city=nyc,
        max_temp=42.0,
        hours_until_close=2.0,
    )

    print("Current Constraints:")
    print(f"\nDallas ({dallas_constraint.hours_until_close:.1f}h until close):")
    print(f"  Max Temp Observed: {dallas_constraint.max_temp_observed}°F")
    print(f"  Observations: {dallas_constraint.observation_count}")
    print(f"  Confidence: {dallas_constraint.confidence:.1%}")
    print(f"  Uncertainty Reduction: {dallas_constraint.uncertainty_reduction:.1%}")

    print(f"\nNYC ({nyc_constraint.hours_until_close:.1f}h until close):")
    print(f"  Max Temp Observed: {nyc_constraint.max_temp_observed}°F")
    print(f"  Observations: {nyc_constraint.observation_count}")
    print(f"  Confidence: {nyc_constraint.confidence:.1%}")
    print(f"  Uncertainty Reduction: {nyc_constraint.uncertainty_reduction:.1%}")

    # Demonstrate constraint application
    print(f"\n{'='*60}")
    print("Applying Constraint to Forecast")
    print(f"{'='*60}\n")

    # Create mock ensemble forecast
    np.random.seed(42)
    forecast_temps = np.concatenate([
        np.random.normal(55, 4, 40),   # ICON
        np.random.normal(56, 4.5, 31), # GFS
        np.random.normal(54, 3.5, 51), # ECMWF
    ])

    print(f"Original Forecast (Dallas):")
    print(f"  Members: {len(forecast_temps)}")
    print(f"  Mean: {np.mean(forecast_temps):.1f}°F")
    print(f"  Std: {np.std(forecast_temps):.1f}°F")
    print(f"  Range: [{np.min(forecast_temps):.1f}, {np.max(forecast_temps):.1f}]°F")

    # Apply constraint
    nowcaster = MetarNowcaster()
    constrained = nowcaster.apply_constraint(forecast_temps, dallas_constraint)

    print(f"\nAfter METAR Constraint (max >= {dallas_constraint.max_temp_observed}°F):")
    print(f"  Members: {len(constrained)}")
    print(f"  Mean: {np.mean(constrained):.1f}°F")
    print(f"  Std: {np.std(constrained):.1f}°F")
    print(f"  Range: [{np.min(constrained):.1f}, {np.max(constrained):.1f}]°F")

    # Show probability impact
    print(f"\n{'='*60}")
    print("Probability Impact")
    print(f"{'='*60}\n")

    buckets = [
        ("<55°F", None, 55.0),
        ("55-58°F", 55.0, 58.0),
        ("58-61°F", 58.0, 61.0),
        ("≥61°F", 61.0, None),
    ]

    print(f"{'Bucket':<12} {'Before':<10} {'After':<10} {'Change':<10}")
    print("-" * 45)

    for name, low, high in buckets:
        # Calculate probability before constraint
        if low is None:
            low_val = float('-inf')
        else:
            low_val = low
        if high is None:
            high_val = float('inf')
        else:
            high_val = high

        before = np.sum((forecast_temps >= low_val) & (forecast_temps < high_val)) / len(forecast_temps)
        after = np.sum((constrained >= low_val) & (constrained < high_val)) / len(constrained)
        change = after - before

        print(f"{name:<12} {before:>8.1%}   {after:>8.1%}   {change:>+8.1%}")

    print(f"\nKey insight: Buckets below observed max ({dallas_constraint.max_temp_observed}°F)")
    print("now have ZERO probability - this is valuable edge!")

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(demo_nowcasting())
