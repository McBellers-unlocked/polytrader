"""Weather Underground historical data client.

Fetches historical temperature data from Wunderground for market resolution.
Polymarket uses Wunderground as the official resolution source.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import aiohttp

from src.logging import get_logger

logger = get_logger(__name__)

# Wunderground base URL for historical data
WUNDERGROUND_BASE_URL = "https://www.wunderground.com/history/daily"


@dataclass
class DailyTemperature:
    """Daily temperature data from Wunderground."""

    station: str
    date: date
    high_temp: float  # In the station's native unit
    low_temp: float
    unit: str  # "F" or "C"
    source_url: str
    fetched_at: datetime


class WundergroundClient:
    """
    Client for fetching historical temperature data from Weather Underground.

    Wunderground is the official resolution source for Polymarket weather markets.
    This client scrapes the historical data page to get the daily high temperature.

    Usage:
        async with WundergroundClient() as client:
            result = await client.get_daily_high(
                wunderground_path="us/wa/seatac/KSEA",
                target_date=date(2026, 1, 25),
            )
            print(f"High temp: {result.high_temp}°{result.unit}")
    """

    def __init__(self, timeout: int = 30):
        """Initialize the client."""
        self._session: aiohttp.ClientSession | None = None
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def __aenter__(self) -> "WundergroundClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Create the HTTP session."""
        if self._session is None:
            # Use headers to look like a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=headers,
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_daily_high(
        self,
        wunderground_path: str,
        target_date: date,
        unit: str = "F",
    ) -> DailyTemperature | None:
        """
        Fetch the daily high temperature from Wunderground.

        Args:
            wunderground_path: Path portion of URL (e.g., "us/wa/seatac/KSEA")
            target_date: The date to fetch data for
            unit: Expected unit ("F" or "C") - used for validation

        Returns:
            DailyTemperature with the high temp, or None if not available
        """
        if not self._session:
            await self.connect()

        # Build the URL
        # Format: https://www.wunderground.com/history/daily/us/wa/seatac/KSEA/date/2026-1-25
        date_str = f"{target_date.year}-{target_date.month}-{target_date.day}"
        url = f"{WUNDERGROUND_BASE_URL}/{wunderground_path}/date/{date_str}"

        logger.debug(
            "Fetching Wunderground data",
            url=url,
            target_date=str(target_date),
        )

        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning(
                        "Wunderground request failed",
                        url=url,
                        status=response.status,
                    )
                    return None

                html = await response.text()

                # Parse the high temperature from the HTML
                high_temp = self._parse_high_temp(html, unit)

                if high_temp is None:
                    logger.warning(
                        "Could not parse high temp from Wunderground",
                        url=url,
                    )
                    return None

                # Also try to get the low temp
                low_temp = self._parse_low_temp(html, unit)

                # Extract station code from path
                station = wunderground_path.split("/")[-1]

                result = DailyTemperature(
                    station=station,
                    date=target_date,
                    high_temp=high_temp,
                    low_temp=low_temp or 0.0,
                    unit=unit,
                    source_url=url,
                    fetched_at=datetime.utcnow(),
                )

                logger.info(
                    "Fetched Wunderground daily high",
                    station=station,
                    date=str(target_date),
                    high_temp=high_temp,
                    unit=unit,
                )

                return result

        except aiohttp.ClientError as e:
            logger.error(
                "Wunderground request error",
                url=url,
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error(
                "Wunderground parsing error",
                url=url,
                error=str(e),
            )
            return None

    def _parse_high_temp(self, html: str, unit: str) -> float | None:
        """
        Parse the high temperature from Wunderground HTML.

        The page has a summary table with Max Temperature.
        We look for patterns like "Max Temperature" followed by a number.
        """
        # Pattern 1: Look for the history summary section
        # Wunderground shows "Max Temperature" with the value
        patterns = [
            # JSON-LD structured data pattern
            r'"temperatureMax":\s*(\d+(?:\.\d+)?)',
            # HTML table pattern - Max Temperature row
            r'Max(?:imum)?\s*Temperature.*?(\d+(?:\.\d+)?)\s*°?' + unit,
            # Alternative pattern with just the number and unit
            r'High:\s*(\d+(?:\.\d+)?)\s*°?' + unit,
            # Data attribute pattern
            r'data-high="(\d+(?:\.\d+)?)"',
            # Another common pattern
            r'class="high"[^>]*>(\d+(?:\.\d+)?)<',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue

        # Fallback: look for temperature values in a reasonable range
        # and take the highest one that looks like a daily high
        temp_pattern = rf'(\d{{1,3}})\s*°\s*{unit}'
        all_temps = re.findall(temp_pattern, html, re.IGNORECASE)

        if all_temps:
            temps = [int(t) for t in all_temps]
            # Filter to reasonable temperature range
            if unit == "F":
                reasonable = [t for t in temps if -40 <= t <= 130]
            else:  # Celsius
                reasonable = [t for t in temps if -40 <= t <= 55]

            if reasonable:
                # Return the maximum as the daily high
                return float(max(reasonable))

        return None

    def _parse_low_temp(self, html: str, unit: str) -> float | None:
        """Parse the low temperature from Wunderground HTML."""
        patterns = [
            r'"temperatureMin":\s*(\d+(?:\.\d+)?)',
            r'Min(?:imum)?\s*Temperature.*?(\d+(?:\.\d+)?)\s*°?' + unit,
            r'Low:\s*(\d+(?:\.\d+)?)\s*°?' + unit,
            r'data-low="(\d+(?:\.\d+)?)"',
            r'class="low"[^>]*>(\d+(?:\.\d+)?)<',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue

        return None


async def demo_wunderground():
    """Demonstrate the Wunderground client."""
    from datetime import timedelta

    print(f"\n{'='*60}")
    print("Wunderground Client Demo")
    print(f"{'='*60}\n")

    # Test with Seattle-Tacoma for yesterday
    yesterday = date.today() - timedelta(days=1)

    async with WundergroundClient() as client:
        result = await client.get_daily_high(
            wunderground_path="us/wa/seatac/KSEA",
            target_date=yesterday,
            unit="F",
        )

        if result:
            print(f"Station: {result.station}")
            print(f"Date: {result.date}")
            print(f"High: {result.high_temp}°{result.unit}")
            print(f"Low: {result.low_temp}°{result.unit}")
            print(f"Source: {result.source_url}")
        else:
            print("Failed to fetch data")

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_wunderground())
