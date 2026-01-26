"""Tomorrow.io API client for ML-enhanced weather forecasts."""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

import aiohttp

from src.config import CityConfig, get_settings
from src.logging import get_logger

logger = get_logger(__name__)

# Cache TTL in seconds - refresh Tomorrow.io data every 180s to stay within free tier (500/day)
# 16 markets × (86400s / 180s) = ~480 calls/day
CACHE_TTL_SECONDS = 180


@dataclass
class TomorrowForecast:
    """
    Tomorrow.io weather forecast with ML post-processing.

    Tomorrow.io uses machine learning to post-process NWP model outputs,
    often providing better short-term accuracy.
    """

    city: CityConfig
    target_date: date
    fetch_time: datetime
    high_temp: float
    low_temp: float
    precipitation_probability: float
    confidence: float  # Model confidence (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "city": self.city.name,
            "target_date": self.target_date.isoformat(),
            "fetch_time": self.fetch_time.isoformat(),
            "high_temp": self.high_temp,
            "low_temp": self.low_temp,
            "precipitation_probability": self.precipitation_probability,
            "confidence": self.confidence,
        }


class TomorrowClient:
    """
    Client for Tomorrow.io API with caching.

    Caches forecasts for 180 seconds to stay within free tier (500 calls/day).
    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize the Tomorrow.io client."""
        self.settings = get_settings()
        self._session = session
        self._owns_session = session is None
        # Cache: {(city_key, target_date_str): (fetch_time, forecast)}
        self._cache: dict[tuple[str, str], tuple[datetime, TomorrowForecast]] = {}

    @property
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.settings.tomorrow_io_api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _get_cache_key(self, city: CityConfig, target_date: date) -> tuple[str, str]:
        """Generate cache key for a forecast."""
        # Use city name as key since CityConfig isn't hashable
        return (city.name, target_date.isoformat())

    def _get_cached(self, city: CityConfig, target_date: date) -> TomorrowForecast | None:
        """Get cached forecast if still valid."""
        key = self._get_cache_key(city, target_date)
        if key in self._cache:
            fetch_time, forecast = self._cache[key]
            age_seconds = (datetime.utcnow() - fetch_time).total_seconds()
            if age_seconds < CACHE_TTL_SECONDS:
                logger.debug(
                    "Using cached Tomorrow.io forecast",
                    city=city.name,
                    target_date=str(target_date),
                    age_seconds=int(age_seconds),
                )
                return forecast
        return None

    def _set_cached(self, city: CityConfig, target_date: date, forecast: TomorrowForecast) -> None:
        """Cache a forecast."""
        key = self._get_cache_key(city, target_date)
        self._cache[key] = (datetime.utcnow(), forecast)

    async def get_forecast(
        self,
        city: CityConfig,
        target_date: date,
    ) -> TomorrowForecast | None:
        """
        Fetch forecast from Tomorrow.io with caching.

        Caches forecasts for 180s to stay within free tier.

        Args:
            city: City configuration
            target_date: The date to forecast

        Returns:
            TomorrowForecast or None if API unavailable
        """
        if not self.is_available:
            logger.debug("Tomorrow.io API key not configured, skipping")
            return None

        # Check cache first
        cached = self._get_cached(city, target_date)
        if cached is not None:
            return cached

        session = await self._get_session()

        # Calculate timesteps needed
        today = datetime.now().date()
        days_ahead = (target_date - today).days

        if days_ahead < 0:
            raise ValueError(f"Target date {target_date} is in the past")
        if days_ahead > 14:
            logger.warning("Tomorrow.io limited to 14-day forecast")
            return None

        try:
            params = {
                "location": f"{city.lat},{city.lon}",
                "fields": [
                    "temperatureMax",
                    "temperatureMin",
                    "precipitationProbability",
                ],
                "timesteps": "1d",
                "units": "metric",  # Always fetch metric, convert later
                "apikey": self.settings.tomorrow_io_api_key,
            }

            url = f"{self.settings.tomorrow_io_url}/timelines"

            async with session.get(url, params=params, timeout=30) as response:
                if response.status == 429:
                    logger.warning("Tomorrow.io rate limit exceeded")
                    return None

                if response.status != 200:
                    text = await response.text()
                    logger.warning(
                        "Tomorrow.io API error",
                        status=response.status,
                        response=text[:200],
                    )
                    return None

                data = await response.json()
                forecast = self._parse_response(data, city, target_date)
                if forecast is not None:
                    self._set_cached(city, target_date, forecast)
                return forecast

        except aiohttp.ClientError as e:
            logger.warning("Tomorrow.io request failed", error=str(e))
            return None

    def _parse_response(
        self,
        data: dict[str, Any],
        city: CityConfig,
        target_date: date,
    ) -> TomorrowForecast | None:
        """Parse Tomorrow.io API response."""
        try:
            timelines = data.get("data", {}).get("timelines", [])
            if not timelines:
                return None

            daily = timelines[0].get("intervals", [])
            target_str = target_date.isoformat()

            for interval in daily:
                interval_date = interval.get("startTime", "")[:10]
                if interval_date == target_str:
                    values = interval.get("values", {})

                    high_temp = values.get("temperatureMax", 0)
                    low_temp = values.get("temperatureMin", 0)

                    # Convert to city's unit
                    if city.unit == "F":
                        high_temp = high_temp * 9 / 5 + 32
                        low_temp = low_temp * 9 / 5 + 32

                    logger.info(
                        "Fetched Tomorrow.io forecast",
                        city=city.name,
                        target_date=str(target_date),
                        high_temp=high_temp,
                        low_temp=low_temp,
                    )

                    return TomorrowForecast(
                        city=city,
                        target_date=target_date,
                        fetch_time=datetime.utcnow(),
                        high_temp=high_temp,
                        low_temp=low_temp,
                        precipitation_probability=values.get(
                            "precipitationProbability", 0
                        ),
                        confidence=0.85,  # Tomorrow.io doesn't expose this directly
                    )

            return None

        except (KeyError, TypeError, IndexError) as e:
            logger.warning("Failed to parse Tomorrow.io response", error=str(e))
            return None
