"""Aviation Weather METAR client for real-time observations."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp

from src.config import CityConfig, get_settings
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetarObservation:
    """
    METAR weather observation from an airport station.

    METAR is the standard format for aviation weather reports,
    providing real-time observations that match the resolution
    source for Polymarket weather markets.
    """

    station: str
    observation_time: datetime
    temperature_c: float
    temperature_f: float
    dewpoint_c: float | None
    wind_speed_kt: int | None
    wind_direction: int | None
    visibility_miles: float | None
    raw_metar: str

    @property
    def age_minutes(self) -> float:
        """Get the age of this observation in minutes."""
        return (datetime.utcnow() - self.observation_time).total_seconds() / 60

    @property
    def is_fresh(self) -> bool:
        """Check if observation is less than 90 minutes old."""
        return self.age_minutes < 90

    def get_temperature(self, unit: str) -> float:
        """Get temperature in the specified unit."""
        if unit == "F":
            return self.temperature_f
        return self.temperature_c

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "station": self.station,
            "observation_time": self.observation_time.isoformat(),
            "temperature_c": self.temperature_c,
            "temperature_f": self.temperature_f,
            "dewpoint_c": self.dewpoint_c,
            "wind_speed_kt": self.wind_speed_kt,
            "wind_direction": self.wind_direction,
            "visibility_miles": self.visibility_miles,
            "age_minutes": self.age_minutes,
            "raw_metar": self.raw_metar,
        }


class MetarClient:
    """
    Client for Aviation Weather METAR API.

    Fetches real-time observations from airport weather stations,
    which are the resolution source for Polymarket weather markets.
    """

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize the METAR client."""
        self.settings = get_settings()
        self._session = session
        self._owns_session = session is None

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

    async def get_observation(self, city: CityConfig) -> MetarObservation | None:
        """
        Fetch current METAR observation for a city.

        Args:
            city: City configuration with METAR station ID

        Returns:
            MetarObservation or None if unavailable
        """
        session = await self._get_session()

        try:
            params = {
                "ids": city.metar,
                "format": "json",
                "taf": "false",
            }

            url = self.settings.aviation_weather_url

            async with session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    logger.warning(
                        "METAR API error",
                        station=city.metar,
                        status=response.status,
                    )
                    return None

                data = await response.json()
                if not data:
                    logger.warning("No METAR data returned", station=city.metar)
                    return None

                return self._parse_metar(data[0], city)

        except aiohttp.ClientError as e:
            logger.warning("METAR request failed", station=city.metar, error=str(e))
            return None

    def _parse_metar(
        self,
        data: dict[str, Any],
        city: CityConfig,
    ) -> MetarObservation | None:
        """Parse METAR JSON response."""
        try:
            # Parse observation time (can be ISO string or Unix timestamp)
            obs_time_raw = data.get("obsTime")
            if obs_time_raw:
                if isinstance(obs_time_raw, (int, float)):
                    # Unix timestamp
                    obs_time = datetime.utcfromtimestamp(obs_time_raw)
                elif isinstance(obs_time_raw, str):
                    # ISO format string
                    obs_time = datetime.fromisoformat(obs_time_raw.replace("Z", "+00:00"))
                else:
                    obs_time = datetime.utcnow()
            else:
                obs_time = datetime.utcnow()

            # Get temperature
            temp_c = data.get("temp")
            if temp_c is None:
                # Try parsing from raw METAR
                raw = data.get("rawOb", "")
                temp_c = self._parse_temp_from_raw(raw)

            if temp_c is None:
                logger.warning("No temperature in METAR", station=city.metar)
                return None

            temp_f = temp_c * 9 / 5 + 32

            observation = MetarObservation(
                station=data.get("icaoId", city.metar),
                observation_time=obs_time,
                temperature_c=temp_c,
                temperature_f=temp_f,
                dewpoint_c=data.get("dewp"),
                wind_speed_kt=data.get("wspd"),
                wind_direction=data.get("wdir"),
                visibility_miles=data.get("visib"),
                raw_metar=data.get("rawOb", ""),
            )

            logger.info(
                "Fetched METAR observation",
                station=city.metar,
                city=city.name,
                temp_c=temp_c,
                temp_f=temp_f,
                age_minutes=observation.age_minutes,
            )

            return observation

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Failed to parse METAR",
                station=city.metar,
                error=str(e),
            )
            return None

    def _parse_temp_from_raw(self, raw_metar: str) -> float | None:
        """
        Parse temperature from raw METAR string.

        METAR format: "temp/dewpoint" where M prefix means negative.
        Example: "24/18" means 24°C temp, 18°C dewpoint
        Example: "M02/M05" means -2°C temp, -5°C dewpoint
        """
        # Pattern: temperature/dewpoint group
        pattern = r"\s(M?\d{2})/(M?\d{2})\s"
        match = re.search(pattern, raw_metar)

        if not match:
            return None

        temp_str = match.group(1)
        if temp_str.startswith("M"):
            return -float(temp_str[1:])
        return float(temp_str)

    async def get_daily_high(
        self,
        city: CityConfig,
        target_date: datetime.date,
    ) -> float | None:
        """
        Get the daily high temperature from historical METAR data.

        This is useful for nowcasting on the target day or getting
        resolution values for past markets.

        Args:
            city: City configuration
            target_date: The date to get the high for

        Returns:
            Daily high temperature in city's unit, or None
        """
        # For current day, just return current temp as approximation
        # A full implementation would fetch historical METARs
        observation = await self.get_observation(city)
        if observation:
            return observation.get_temperature(city.unit)
        return None
