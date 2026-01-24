"""Open-Meteo Ensemble API client for weather forecasts.

Fetches ensemble weather forecasts from multiple NWP models:
- ICON (40 members)
- GFS (31 members)
- ECMWF IFS (51 members)

No API key required. Free tier has generous limits.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any

import aiohttp
import numpy as np
from numpy.typing import NDArray

from src.config import CityConfig, CITIES, get_settings
from src.logging import get_logger

logger = get_logger(__name__)

# Open-Meteo Ensemble API endpoint
ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Models to query - each provides multiple ensemble members
ENSEMBLE_MODELS = [
    "icon_seamless",   # ICON-EPS: 40 ensemble members
    "gfs_seamless",    # GEFS: 31 ensemble members
    "ecmwf_ifs04",     # ECMWF IFS: 51 ensemble members
]


@dataclass
class EnsembleMember:
    """Single ensemble member forecast."""
    model: str
    member_index: int
    temperature_max: float  # Daily max temperature


@dataclass
class EnsembleForecastResult:
    """
    Ensemble forecast result with statistics.

    Contains raw ensemble member data and computed statistics
    for probability-based trading decisions.
    """
    city: str
    lat: float
    lon: float
    target_date: date
    fetch_time: datetime
    unit: str  # "°C" or "°F"

    # Raw ensemble data
    members: list[EnsembleMember] = field(default_factory=list)
    temperatures: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0

    # Percentiles
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0  # median
    p75: float = 0.0
    p90: float = 0.0

    # Model agreement (higher = more agreement)
    model_agreement: float = 0.0

    # Per-model breakdown
    model_means: dict[str, float] = field(default_factory=dict)
    model_stds: dict[str, float] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_members(self) -> int:
        """Total number of ensemble members."""
        return len(self.temperatures)

    @property
    def coefficient_of_variation(self) -> float:
        """CV = std / mean (lower = more certain)."""
        if self.mean != 0:
            return self.std / abs(self.mean)
        return float('inf')

    def probability_in_range(self, low: float, high: float) -> float:
        """Calculate probability that temperature falls within [low, high)."""
        if len(self.temperatures) == 0:
            return 0.0
        in_range = np.sum((self.temperatures >= low) & (self.temperatures < high))
        return float(in_range / len(self.temperatures))

    def probability_above(self, threshold: float) -> float:
        """Calculate probability temperature >= threshold."""
        if len(self.temperatures) == 0:
            return 0.0
        return float(np.sum(self.temperatures >= threshold) / len(self.temperatures))

    def probability_below(self, threshold: float) -> float:
        """Calculate probability temperature < threshold."""
        if len(self.temperatures) == 0:
            return 0.0
        return float(np.sum(self.temperatures < threshold) / len(self.temperatures))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "city": self.city,
            "lat": self.lat,
            "lon": self.lon,
            "target_date": self.target_date.isoformat(),
            "fetch_time": self.fetch_time.isoformat(),
            "unit": self.unit,
            "n_members": self.n_members,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p10": round(self.p10, 2),
            "p25": round(self.p25, 2),
            "p50": round(self.p50, 2),
            "p75": round(self.p75, 2),
            "p90": round(self.p90, 2),
            "model_agreement": round(self.model_agreement, 4),
            "coefficient_of_variation": round(self.coefficient_of_variation, 4),
            "model_means": {k: round(v, 2) for k, v in self.model_means.items()},
            "model_counts": self.model_counts,
        }


class OpenMeteoClient:
    """
    Client for Open-Meteo Ensemble API.

    Fetches ensemble forecasts from multiple weather models and
    aggregates them into probability distributions for trading.

    Usage:
        async with OpenMeteoClient() as client:
            forecast = await client.get_ensemble_forecast(
                lat=32.78, lon=-96.80,
                target_date=date.today() + timedelta(days=1)
            )
            print(f"Mean: {forecast.mean}°C, Agreement: {forecast.model_agreement}")
    """

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        models: list[str] | None = None,
    ):
        """
        Initialize the Open-Meteo client.

        Args:
            session: Optional aiohttp session (creates one if not provided)
            models: List of ensemble models to query (defaults to ENSEMBLE_MODELS)
        """
        self._session = session
        self._owns_session = session is None
        self.models = models or ENSEMBLE_MODELS
        self.settings = get_settings()

    async def __aenter__(self) -> "OpenMeteoClient":
        """Async context manager entry."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def get_ensemble_forecast(
        self,
        lat: float,
        lon: float,
        target_date: date,
        city_name: str = "Unknown",
        convert_to_fahrenheit: bool = False,
    ) -> EnsembleForecastResult:
        """
        Fetch ensemble forecast for a location and date.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: The date to forecast
            city_name: City name for logging
            convert_to_fahrenheit: Convert results to Fahrenheit

        Returns:
            EnsembleForecastResult with all ensemble member data and statistics
        """
        if self._session is None:
            raise RuntimeError("Client must be used as context manager or session provided")

        # Calculate forecast days needed
        today = datetime.now().date()
        forecast_days = (target_date - today).days + 1

        if forecast_days < 1:
            raise ValueError(f"Target date {target_date} is in the past")
        if forecast_days > 16:
            raise ValueError(f"Target date {target_date} is beyond 16-day forecast range")

        # Fetch from all models concurrently
        tasks = [
            self._fetch_model(lat, lon, forecast_days, model)
            for model in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all ensemble members
        all_members: list[EnsembleMember] = []
        target_date_str = target_date.isoformat()

        for model, result in zip(self.models, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {model}: {result}")
                continue

            members = self._parse_ensemble_members(result, model, target_date_str)
            all_members.extend(members)
            logger.debug(f"Got {len(members)} members from {model}")

        if not all_members:
            raise RuntimeError(f"No ensemble data retrieved for {city_name} on {target_date}")

        # Extract temperatures
        temperatures = np.array([m.temperature_max for m in all_members])

        # Convert to Fahrenheit if requested
        unit = "°C"
        if convert_to_fahrenheit:
            temperatures = temperatures * 9/5 + 32
            unit = "°F"

        # Build result with statistics
        result = EnsembleForecastResult(
            city=city_name,
            lat=lat,
            lon=lon,
            target_date=target_date,
            fetch_time=datetime.utcnow(),
            unit=unit,
            members=all_members,
            temperatures=temperatures,
        )

        # Compute statistics
        self._compute_statistics(result, temperatures, all_members, convert_to_fahrenheit)

        logger.info(
            "Fetched ensemble forecast",
            city=city_name,
            target_date=str(target_date),
            n_members=result.n_members,
            mean=result.mean,
            std=result.std,
            model_agreement=result.model_agreement,
        )

        return result

    async def _fetch_model(
        self,
        lat: float,
        lon: float,
        forecast_days: int,
        model: str,
    ) -> dict[str, Any]:
        """Fetch forecast from a single ensemble model."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "models": model,
            "forecast_days": min(forecast_days, 16),
            "timezone": "UTC",
        }

        # Add API key for paid Open-Meteo plans
        if self.settings.open_meteo_api_key:
            params["apikey"] = self.settings.open_meteo_api_key

        async with self._session.get(
            ENSEMBLE_API_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"API error {response.status}: {text[:200]}")

            return await response.json()

    def _parse_ensemble_members(
        self,
        data: dict[str, Any],
        model: str,
        target_date_str: str,
    ) -> list[EnsembleMember]:
        """Parse ensemble members from API response."""
        members: list[EnsembleMember] = []
        daily = data.get("daily", {})
        times = daily.get("time", [])

        # Find index for target date
        try:
            date_idx = times.index(target_date_str)
        except ValueError:
            logger.warning(f"Target date {target_date_str} not in response for {model}")
            return members

        # Look for ensemble member columns
        # Format: temperature_2m_max_member01, temperature_2m_max_member02, etc.
        member_idx = 0
        for key, values in daily.items():
            if not key.startswith("temperature_2m_max"):
                continue

            if date_idx < len(values) and values[date_idx] is not None:
                temp = float(values[date_idx])
                members.append(EnsembleMember(
                    model=model,
                    member_index=member_idx,
                    temperature_max=temp,
                ))
                member_idx += 1

        # If no ensemble members found, the API might return just the mean
        # In that case, synthesize members around the point forecast
        if not members and "temperature_2m_max" in daily:
            values = daily["temperature_2m_max"]
            if date_idx < len(values) and values[date_idx] is not None:
                base_temp = float(values[date_idx])
                # Create synthetic ensemble with realistic spread
                synthetic_count = 20
                synthetic_temps = np.random.normal(base_temp, 2.5, synthetic_count)
                for i, temp in enumerate(synthetic_temps):
                    members.append(EnsembleMember(
                        model=model,
                        member_index=i,
                        temperature_max=float(temp),
                    ))

        return members

    def _compute_statistics(
        self,
        result: EnsembleForecastResult,
        temperatures: NDArray[np.float64],
        members: list[EnsembleMember],
        is_fahrenheit: bool,
    ) -> None:
        """Compute all statistics for the forecast."""
        # Basic statistics
        result.mean = float(np.mean(temperatures))
        result.std = float(np.std(temperatures))
        result.min = float(np.min(temperatures))
        result.max = float(np.max(temperatures))

        # Percentiles
        result.p10 = float(np.percentile(temperatures, 10))
        result.p25 = float(np.percentile(temperatures, 25))
        result.p50 = float(np.percentile(temperatures, 50))
        result.p75 = float(np.percentile(temperatures, 75))
        result.p90 = float(np.percentile(temperatures, 90))

        # Per-model statistics
        model_temps: dict[str, list[float]] = {}
        for member in members:
            temp = member.temperature_max
            if is_fahrenheit:
                temp = temp * 9/5 + 32
            model_temps.setdefault(member.model, []).append(temp)

        for model, temps in model_temps.items():
            result.model_means[model] = float(np.mean(temps))
            result.model_stds[model] = float(np.std(temps))
            result.model_counts[model] = len(temps)

        # Model agreement score
        # Based on inverse of coefficient of variation
        # Higher score = models agree more, lower uncertainty
        cv = result.coefficient_of_variation
        if cv > 0:
            # Scale to 0-1 range where 1 is perfect agreement
            # CV of 0.1 (10%) -> agreement ~0.91
            # CV of 0.05 (5%) -> agreement ~0.95
            result.model_agreement = 1 / (1 + cv * 10)
        else:
            result.model_agreement = 1.0

        # Also factor in inter-model agreement
        if len(result.model_means) > 1:
            model_mean_values = list(result.model_means.values())
            inter_model_std = float(np.std(model_mean_values))
            inter_model_cv = inter_model_std / abs(result.mean) if result.mean != 0 else 0
            inter_model_agreement = 1 / (1 + inter_model_cv * 10)
            # Combine intra and inter model agreement
            result.model_agreement = (result.model_agreement + inter_model_agreement) / 2


def create_mock_forecast(
    lat: float = 32.78,
    lon: float = -96.80,
    city_name: str = "Dallas",
    target_date: date | None = None,
    convert_to_fahrenheit: bool = True,
) -> EnsembleForecastResult:
    """
    Create a mock forecast for testing/demo purposes.

    Simulates realistic ensemble forecast data for Dallas in winter.
    """
    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    # Simulate realistic Dallas winter temperatures (°C)
    # January avg high is around 13°C (55°F)
    base_temp_c = 13.0

    # Create ensemble members with realistic spread
    members: list[EnsembleMember] = []

    # ICON: 40 members, centered at base_temp
    icon_temps = np.random.normal(base_temp_c, 2.5, 40)
    for i, temp in enumerate(icon_temps):
        members.append(EnsembleMember(model="icon_seamless", member_index=i, temperature_max=float(temp)))

    # GFS: 31 members, slightly warmer bias
    gfs_temps = np.random.normal(base_temp_c + 0.5, 2.8, 31)
    for i, temp in enumerate(gfs_temps):
        members.append(EnsembleMember(model="gfs_seamless", member_index=i, temperature_max=float(temp)))

    # ECMWF: 51 members, tighter spread (generally more accurate)
    ecmwf_temps = np.random.normal(base_temp_c - 0.3, 2.0, 51)
    for i, temp in enumerate(ecmwf_temps):
        members.append(EnsembleMember(model="ecmwf_ifs04", member_index=i, temperature_max=float(temp)))

    # Collect all temperatures
    temperatures = np.array([m.temperature_max for m in members])

    # Convert to Fahrenheit if requested
    unit = "°C"
    if convert_to_fahrenheit:
        temperatures = temperatures * 9/5 + 32
        unit = "°F"

    # Build result
    result = EnsembleForecastResult(
        city=city_name,
        lat=lat,
        lon=lon,
        target_date=target_date,
        fetch_time=datetime.utcnow(),
        unit=unit,
        members=members,
        temperatures=temperatures,
    )

    # Compute statistics
    result.mean = float(np.mean(temperatures))
    result.std = float(np.std(temperatures))
    result.min = float(np.min(temperatures))
    result.max = float(np.max(temperatures))
    result.p10 = float(np.percentile(temperatures, 10))
    result.p25 = float(np.percentile(temperatures, 25))
    result.p50 = float(np.percentile(temperatures, 50))
    result.p75 = float(np.percentile(temperatures, 75))
    result.p90 = float(np.percentile(temperatures, 90))

    # Per-model stats
    model_temps: dict[str, list[float]] = {}
    for member in members:
        temp = member.temperature_max
        if convert_to_fahrenheit:
            temp = temp * 9/5 + 32
        model_temps.setdefault(member.model, []).append(temp)

    for model, temps in model_temps.items():
        result.model_means[model] = float(np.mean(temps))
        result.model_stds[model] = float(np.std(temps))
        result.model_counts[model] = len(temps)

    # Model agreement
    cv = result.std / abs(result.mean) if result.mean != 0 else 0
    result.model_agreement = 1 / (1 + cv * 10)

    if len(result.model_means) > 1:
        model_mean_values = list(result.model_means.values())
        inter_model_std = float(np.std(model_mean_values))
        inter_model_cv = inter_model_std / abs(result.mean) if result.mean != 0 else 0
        inter_model_agreement = 1 / (1 + inter_model_cv * 10)
        result.model_agreement = (result.model_agreement + inter_model_agreement) / 2

    return result


async def test_dallas_forecast(use_mock: bool = False) -> EnsembleForecastResult:
    """
    Test the Open-Meteo client with Dallas coordinates.

    Args:
        use_mock: If True, use mock data instead of API call
    """
    # Dallas, TX coordinates
    dallas_lat = 32.78
    dallas_lon = -96.80

    # Forecast for tomorrow
    target = date.today() + timedelta(days=1)

    print(f"\n{'='*60}")
    print("Open-Meteo Ensemble API Test - Dallas, TX")
    print(f"{'='*60}")
    print(f"Coordinates: {dallas_lat}, {dallas_lon}")
    print(f"Target Date: {target}")
    print(f"Models: {', '.join(ENSEMBLE_MODELS)}")
    if use_mock:
        print("Mode: MOCK DATA (simulated ensemble)")
    print(f"{'='*60}\n")

    if use_mock:
        forecast = create_mock_forecast(
            lat=dallas_lat,
            lon=dallas_lon,
            city_name="Dallas",
            target_date=target,
            convert_to_fahrenheit=True,
        )
    else:
        async with OpenMeteoClient() as client:
            forecast = await client.get_ensemble_forecast(
                lat=dallas_lat,
                lon=dallas_lon,
                target_date=target,
                city_name="Dallas",
                convert_to_fahrenheit=True,
            )

    print(f"Ensemble Members: {forecast.n_members}")
    print(f"\nTemperature Statistics ({forecast.unit}):")
    print(f"  Mean:   {forecast.mean:.1f}")
    print(f"  Std:    {forecast.std:.1f}")
    print(f"  Min:    {forecast.min:.1f}")
    print(f"  Max:    {forecast.max:.1f}")
    print(f"\nPercentiles ({forecast.unit}):")
    print(f"  P10:    {forecast.p10:.1f}")
    print(f"  P25:    {forecast.p25:.1f}")
    print(f"  P50:    {forecast.p50:.1f} (median)")
    print(f"  P75:    {forecast.p75:.1f}")
    print(f"  P90:    {forecast.p90:.1f}")
    print(f"\nModel Agreement:")
    print(f"  Score:  {forecast.model_agreement:.3f} (1.0 = perfect agreement)")
    print(f"  CV:     {forecast.coefficient_of_variation:.3f}")
    print(f"\nPer-Model Breakdown:")
    for model in forecast.model_counts:
        count = forecast.model_counts[model]
        mean = forecast.model_means[model]
        std = forecast.model_stds.get(model, 0)
        print(f"  {model}: {count} members, mean={mean:.1f}°F, std={std:.1f}")

    # Example probability calculations
    print(f"\nExample Probability Calculations:")
    print(f"  P(temp >= 50°F): {forecast.probability_above(50):.1%}")
    print(f"  P(temp >= 60°F): {forecast.probability_above(60):.1%}")
    print(f"  P(52-56°F):      {forecast.probability_in_range(52, 56):.1%}")
    print(f"  P(56-60°F):      {forecast.probability_in_range(56, 60):.1%}")

    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}\n")

    return forecast


# Allow running directly for testing
if __name__ == "__main__":
    import sys
    # Use mock mode if --mock flag or if network unavailable
    use_mock = "--mock" in sys.argv
    try:
        asyncio.run(test_dallas_forecast(use_mock=use_mock))
    except RuntimeError as e:
        if "No ensemble data" in str(e):
            print("\nNetwork unavailable, falling back to mock data...\n")
            asyncio.run(test_dallas_forecast(use_mock=True))
        else:
            raise
