"""Polymarket weather market scanner."""

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import aiohttp

from src.config import CityConfig, CITIES, get_settings
from src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemperatureBucket:
    """
    A temperature bucket in a weather market.

    Each bucket represents a range like "68-69°F" with YES/NO tokens.
    """

    token_id: str
    outcome: str  # e.g., "68-69°F" or "≥75°F" or "<60°F"
    low_bound: float | None  # Lower bound (inclusive), None for unbounded
    high_bound: float | None  # Upper bound (exclusive), None for unbounded
    yes_price: float  # Current YES price (0-1)
    no_price: float  # Current NO price (0-1)
    yes_bid: float  # Best bid for YES
    yes_ask: float  # Best ask for YES
    no_bid: float  # Best bid for NO
    no_ask: float  # Best ask for NO
    volume_24h: float = 0.0  # 24h trading volume

    @property
    def spread(self) -> float:
        """Get bid-ask spread for YES token."""
        return self.yes_ask - self.yes_bid

    @property
    def midpoint(self) -> float:
        """Get midpoint price for YES token."""
        return (self.yes_bid + self.yes_ask) / 2

    def contains_temp(self, temp: float) -> bool:
        """Check if a temperature falls within this bucket."""
        if self.low_bound is not None and temp < self.low_bound:
            return False
        if self.high_bound is not None and temp >= self.high_bound:
            return False
        return True


@dataclass
class WeatherMarket:
    """
    A Polymarket weather prediction market.

    Represents a market like "Highest temperature in NYC on Jan 25"
    with multiple temperature bucket outcomes.
    """

    condition_id: str
    question: str
    description: str
    city: CityConfig | None
    target_date: date | None
    end_date: datetime  # Market close time
    buckets: list[TemperatureBucket] = field(default_factory=list)
    total_volume: float = 0.0
    liquidity: float = 0.0

    @property
    def is_active(self) -> bool:
        """Check if market is still active."""
        return datetime.utcnow() < self.end_date

    @property
    def hours_until_close(self) -> float:
        """Hours until market closes."""
        delta = self.end_date - datetime.utcnow()
        return delta.total_seconds() / 3600

    @property
    def total_yes_prices(self) -> float:
        """Sum of all YES prices (should be close to 1.0)."""
        return sum(b.yes_price for b in self.buckets)

    @property
    def arbitrage_gap(self) -> float:
        """Gap from 1.0 (positive = buy opportunity, negative = sell opportunity)."""
        return 1.0 - self.total_yes_prices

    def get_bucket_for_temp(self, temp: float) -> TemperatureBucket | None:
        """Find the bucket that contains a given temperature."""
        for bucket in self.buckets:
            if bucket.contains_temp(temp):
                return bucket
        return None


class MarketScanner:
    """
    Scans Polymarket for active weather prediction markets.

    Uses the Gamma API to discover markets and the CLOB API for orderbooks.
    """

    # Keywords to identify weather markets
    WEATHER_KEYWORDS = [
        "temperature",
        "highest temp",
        "weather",
        "degrees",
        "°F",
        "°C",
    ]

    # Regex patterns for parsing market questions
    CITY_PATTERNS = {
        "nyc": [r"new york", r"nyc", r"manhattan"],
        "london": [r"london"],
        "seoul": [r"seoul"],
        "dallas": [r"dallas"],
        "toronto": [r"toronto"],
        "seattle": [r"seattle"],
        "atlanta": [r"atlanta"],
    }

    DATE_PATTERN = re.compile(
        r"(?:on\s+)?(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s*(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(?P<year>\d{4}))?",
        re.IGNORECASE,
    )

    BUCKET_PATTERN = re.compile(
        r"(?P<type>≥|>=|>|≤|<=|<)?\s*(?P<low>\d+(?:\.\d+)?)\s*"
        r"(?:°?[FC])?\s*(?:-|to)?\s*(?P<high>\d+(?:\.\d+)?)?\s*(?:°?[FC])?",
        re.IGNORECASE,
    )

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize the market scanner."""
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

    async def scan_weather_markets(self) -> list[WeatherMarket]:
        """
        Scan for active weather prediction markets.

        Returns:
            List of WeatherMarket objects
        """
        session = await self._get_session()
        markets: list[WeatherMarket] = []

        try:
            # Fetch markets from Gamma API
            url = f"{self.settings.polymarket_gamma_url}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
            }

            async with session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    logger.error(
                        "Failed to fetch markets from Gamma API",
                        status=response.status,
                    )
                    return markets

                data = await response.json()

                for market_data in data:
                    market = self._parse_market(market_data)
                    if market and market.city:
                        markets.append(market)

            logger.info(
                "Scanned weather markets",
                total_markets=len(data) if isinstance(data, list) else 0,
                weather_markets=len(markets),
            )

        except aiohttp.ClientError as e:
            logger.error("Failed to scan markets", error=str(e))

        return markets

    async def get_market_details(
        self,
        condition_id: str,
    ) -> WeatherMarket | None:
        """
        Get detailed information for a specific market.

        Args:
            condition_id: The market condition ID

        Returns:
            WeatherMarket with full bucket details
        """
        session = await self._get_session()

        try:
            # Fetch market details
            url = f"{self.settings.polymarket_gamma_url}/markets/{condition_id}"

            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                return self._parse_market(data)

        except aiohttp.ClientError as e:
            logger.error(
                "Failed to fetch market details",
                condition_id=condition_id,
                error=str(e),
            )
            return None

    async def get_orderbook(
        self,
        token_id: str,
    ) -> dict[str, Any]:
        """
        Get orderbook for a specific token.

        Args:
            token_id: The token ID

        Returns:
            Orderbook data with bids and asks
        """
        session = await self._get_session()

        try:
            url = f"{self.settings.polymarket_clob_url}/book"
            params = {"token_id": token_id}

            async with session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    return {"bids": [], "asks": []}

                return await response.json()

        except aiohttp.ClientError as e:
            logger.warning("Failed to fetch orderbook", token_id=token_id, error=str(e))
            return {"bids": [], "asks": []}

    def _parse_market(self, data: dict[str, Any]) -> WeatherMarket | None:
        """Parse market data from Gamma API response."""
        try:
            question = data.get("question", "")
            description = data.get("description", "")

            # Check if this is a weather market
            combined_text = (question + " " + description).lower()
            if not any(kw in combined_text for kw in self.WEATHER_KEYWORDS):
                return None

            # Parse city
            city = self._parse_city(combined_text)
            if not city:
                return None

            # Parse target date
            target_date = self._parse_date(combined_text)

            # Parse end date
            end_date_str = data.get("endDate") or data.get("end_date_iso")
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            else:
                end_date = datetime.utcnow()

            # Parse buckets from outcomes
            buckets = self._parse_buckets(data.get("tokens", []))

            market = WeatherMarket(
                condition_id=data.get("conditionId", data.get("condition_id", "")),
                question=question,
                description=description,
                city=city,
                target_date=target_date,
                end_date=end_date,
                buckets=buckets,
                total_volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
            )

            logger.debug(
                "Parsed weather market",
                condition_id=market.condition_id,
                city=city.name if city else None,
                target_date=str(target_date) if target_date else None,
                n_buckets=len(buckets),
            )

            return market

        except Exception as e:
            logger.warning("Failed to parse market", error=str(e))
            return None

    def _parse_city(self, text: str) -> CityConfig | None:
        """Extract city from market text."""
        text_lower = text.lower()
        for city_key, patterns in self.CITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return CITIES.get(city_key)
        return None

    def _parse_date(self, text: str) -> date | None:
        """Extract target date from market text."""
        match = self.DATE_PATTERN.search(text)
        if not match:
            return None

        month_str = match.group("month").lower()[:3]
        day = int(match.group("day"))
        year = int(match.group("year")) if match.group("year") else datetime.now().year

        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month = month_map.get(month_str, 1)

        try:
            return date(year, month, day)
        except ValueError:
            return None

    def _parse_buckets(
        self,
        tokens: list[dict[str, Any]],
    ) -> list[TemperatureBucket]:
        """Parse temperature buckets from token data."""
        buckets: list[TemperatureBucket] = []

        for token in tokens:
            outcome = token.get("outcome", "")
            token_id = token.get("token_id", "")

            # Parse bounds from outcome string
            low_bound, high_bound = self._parse_bounds(outcome)

            # Get prices
            yes_price = float(token.get("price", 0) or 0)

            bucket = TemperatureBucket(
                token_id=token_id,
                outcome=outcome,
                low_bound=low_bound,
                high_bound=high_bound,
                yes_price=yes_price,
                no_price=1 - yes_price,
                yes_bid=yes_price * 0.98,  # Estimated from price
                yes_ask=yes_price * 1.02,
                no_bid=(1 - yes_price) * 0.98,
                no_ask=(1 - yes_price) * 1.02,
                volume_24h=float(token.get("volume", 0) or 0),
            )

            buckets.append(bucket)

        # Sort by low bound
        buckets.sort(key=lambda b: b.low_bound if b.low_bound is not None else float("-inf"))

        return buckets

    def _parse_bounds(self, outcome: str) -> tuple[float | None, float | None]:
        """Parse temperature bounds from outcome string."""
        outcome = outcome.strip()

        # Handle special cases
        if outcome.startswith(("≥", ">=", ">")):
            match = re.search(r"(\d+(?:\.\d+)?)", outcome)
            if match:
                return float(match.group(1)), None

        if outcome.startswith(("≤", "<=", "<")):
            match = re.search(r"(\d+(?:\.\d+)?)", outcome)
            if match:
                return None, float(match.group(1))

        # Handle range like "68-69" or "68-69°F"
        range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", outcome)
        if range_match:
            return float(range_match.group(1)), float(range_match.group(2))

        # Single value
        single_match = re.search(r"(\d+(?:\.\d+)?)", outcome)
        if single_match:
            val = float(single_match.group(1))
            return val, val + 1  # Assume 1-degree bucket

        return None, None
