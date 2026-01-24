"""Polymarket weather market scanner.

Scans for active weather prediction markets and parses temperature buckets
for trading strategy integration.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any

import aiohttp

from src.config import CityConfig, CITIES, get_settings
from src.logging import get_logger

logger = get_logger(__name__)

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_API_URL = "https://clob.polymarket.com"


@dataclass
class TemperatureBucket:
    """
    A temperature bucket/outcome in a weather market.

    Examples: "68-69°F", "≥75°F", "<60°F"
    """
    token_id: str  # YES token ID
    outcome_id: str
    outcome: str  # Raw string like "68-69°F"

    # Parsed bounds
    low_bound: float | None = None  # Inclusive lower bound
    high_bound: float | None = None  # Exclusive upper bound
    unit: str = "F"  # "F" or "C"

    # Pricing (0-1 scale, representing probability)
    yes_price: float = 0.0
    no_price: float = 0.0

    # NO token for BUY NO trades
    no_token_id: str = ""  # NO token ID (for neg-risk markets)

    # Orderbook data
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    spread: float = 0.0

    # Volume
    volume_24h: float = 0.0

    @property
    def midpoint(self) -> float:
        """Midpoint price between bid and ask."""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.yes_price

    @property
    def is_bounded(self) -> bool:
        """Check if this is a bounded range (not open-ended)."""
        return self.low_bound is not None and self.high_bound is not None

    @property
    def range_width(self) -> float | None:
        """Width of the temperature range in degrees."""
        if self.is_bounded:
            return self.high_bound - self.low_bound
        return None

    @property
    def range_midpoint(self) -> float | None:
        """Midpoint temperature of the bucket."""
        if self.is_bounded:
            return (self.low_bound + self.high_bound) / 2
        return None

    def contains_temp(self, temp: float) -> bool:
        """Check if a temperature falls within this bucket."""
        if self.low_bound is not None and temp < self.low_bound:
            return False
        if self.high_bound is not None and temp >= self.high_bound:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "outcome": self.outcome,
            "low_bound": self.low_bound,
            "high_bound": self.high_bound,
            "unit": self.unit,
            "yes_price": self.yes_price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "midpoint": self.midpoint,
        }


@dataclass
class WeatherMarket:
    """
    A Polymarket weather prediction market.

    Represents a "Highest temperature in [CITY] on [DATE]" market
    with multiple temperature bucket outcomes.
    """
    condition_id: str
    question_id: str
    question: str
    description: str
    market_slug: str

    # Mapped city
    city: CityConfig | None = None
    city_key: str = ""

    # Target date parsed from question
    target_date: date | None = None

    # Market timing
    end_date: datetime | None = None
    created_at: datetime | None = None

    # Temperature buckets
    buckets: list[TemperatureBucket] = field(default_factory=list)

    # Market stats
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0

    # Resolution
    resolution_source: str = ""
    is_resolved: bool = False

    @property
    def is_active(self) -> bool:
        """Check if market is still active for trading."""
        if self.is_resolved:
            return False
        if self.end_date:
            now = datetime.now(timezone.utc)
            # Handle both naive and aware datetimes
            end = self.end_date
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if now >= end:
                return False
        return True

    @property
    def hours_until_close(self) -> float:
        """Hours remaining until market closes."""
        if self.end_date:
            now = datetime.now(timezone.utc)
            # Handle both naive and aware datetimes
            end = self.end_date
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            delta = end - now
            return max(0, delta.total_seconds() / 3600)
        return float('inf')

    @property
    def total_yes_prices(self) -> float:
        """Sum of all YES prices (should be ~1.0 in efficient market)."""
        return sum(b.yes_price for b in self.buckets)

    @property
    def arbitrage_gap(self) -> float:
        """Gap from 1.0 - positive means buying opportunity."""
        return 1.0 - self.total_yes_prices

    @property
    def n_buckets(self) -> int:
        """Number of temperature buckets."""
        return len(self.buckets)

    def get_bucket_for_temp(self, temp: float) -> TemperatureBucket | None:
        """Find the bucket containing a given temperature."""
        for bucket in self.buckets:
            if bucket.contains_temp(temp):
                return bucket
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "city": self.city.name if self.city else None,
            "city_key": self.city_key,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_active": self.is_active,
            "hours_until_close": self.hours_until_close,
            "n_buckets": self.n_buckets,
            "total_yes_prices": self.total_yes_prices,
            "arbitrage_gap": self.arbitrage_gap,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "buckets": [b.to_dict() for b in self.buckets],
        }


class MarketScanner:
    """
    Scans Polymarket for active weather prediction markets.

    Uses the Gamma API to discover markets and CLOB API for orderbook data.

    Usage:
        async with MarketScanner() as scanner:
            markets = await scanner.scan_weather_markets()
            for market in markets:
                print(f"{market.city.name}: {market.n_buckets} buckets")
    """

    # Patterns for city detection
    CITY_PATTERNS: dict[str, list[str]] = {
        "nyc": [r"new york", r"\bnyc\b", r"manhattan"],
        "london": [r"\blondon\b"],
        "seoul": [r"\bseoul\b"],
        "dallas": [r"\bdallas\b"],
        "toronto": [r"\btoronto\b"],
        "seattle": [r"\bseattle\b"],
        "atlanta": [r"\batlanta\b"],
        "chicago": [r"\bchicago\b"],
        "miami": [r"\bmiami\b"],
        "la": [r"los angeles", r"\bla\b"],
        "phoenix": [r"\bphoenix\b"],
        "denver": [r"\bdenver\b"],
    }

    # City key to slug name mapping (for generating API slugs)
    CITY_SLUG_NAMES: dict[str, str] = {
        "nyc": "new-york-city",
        "london": "london",
        "seoul": "seoul",
        "dallas": "dallas",
        "toronto": "toronto",
        "seattle": "seattle",
        "atlanta": "atlanta",
        "chicago": "chicago",
        "miami": "miami",
        "la": "los-angeles",
        "phoenix": "phoenix",
        "denver": "denver",
    }

    # Date patterns
    DATE_PATTERNS = [
        # "January 25" or "Jan 25"
        re.compile(
            r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(?P<year>\d{4}))?",
            re.IGNORECASE,
        ),
        # "1/25" or "01/25/2026"
        re.compile(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?"),
    ]

    # Temperature bucket patterns
    BUCKET_PATTERNS = [
        # Range: "68-69°F" or "68-69 F" or "68 to 69"
        re.compile(r"(?P<low>\d+(?:\.\d+)?)\s*(?:°?[FC])?\s*[-–to]+\s*(?P<high>\d+(?:\.\d+)?)\s*°?(?P<unit>[FC])?", re.IGNORECASE),
        # Greater than: "≥75°F" or ">= 75" or "75+"
        re.compile(r"(?:≥|>=|>|over)\s*(?P<threshold>\d+(?:\.\d+)?)\s*°?(?P<unit>[FC])?", re.IGNORECASE),
        re.compile(r"(?P<threshold>\d+(?:\.\d+)?)\s*°?(?P<unit>[FC])?\s*(?:\+|or more|or higher)", re.IGNORECASE),
        # Less than: "<60°F" or "under 60"
        re.compile(r"(?:≤|<=|<|under|below)\s*(?P<threshold>\d+(?:\.\d+)?)\s*°?(?P<unit>[FC])?", re.IGNORECASE),
    ]

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize the market scanner."""
        self.settings = get_settings()
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "MarketScanner":
        """Async context manager entry."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def scan_weather_markets(
        self,
        include_inactive: bool = False,
        fetch_orderbooks: bool = True,
    ) -> list[WeatherMarket]:
        """
        Scan for active weather prediction markets.

        Args:
            include_inactive: Include closed/resolved markets
            fetch_orderbooks: Fetch detailed orderbook data from CLOB

        Returns:
            List of WeatherMarket objects
        """
        if self._session is None:
            raise RuntimeError("Scanner must be used as context manager")

        logger.info("Scanning for weather markets")

        # Fetch events (not individual markets) from Gamma API
        events = await self._fetch_gamma_events()

        # Parse each event into a WeatherMarket with temperature buckets
        weather_markets: list[WeatherMarket] = []
        for event_data in events:
            market = self._parse_event_to_market(event_data)
            if market is None:
                continue

            # Filter for valid temperature markets
            if not self._is_temperature_market(market):
                continue

            # Skip inactive unless requested
            if not include_inactive and not market.is_active:
                continue

            weather_markets.append(market)

        logger.info(f"Found {len(weather_markets)} temperature markets")

        # Fetch orderbook data for each market
        if fetch_orderbooks and weather_markets:
            await self._fetch_orderbooks(weather_markets)

        # Sort by hours until close
        weather_markets.sort(key=lambda m: m.hours_until_close)

        return weather_markets

    def _generate_event_slugs(self, days_ahead: int = 7) -> list[tuple[str, str, date]]:
        """
        Generate event slugs for temperature markets.

        Returns list of (slug, city_key, target_date) tuples.
        """
        from datetime import timedelta

        slugs = []
        today = date.today()

        # Month names for slug generation
        month_names = [
            "", "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]

        for city_key, city_slug in self.CITY_SLUG_NAMES.items():
            # Check that city is in our CITIES config
            if city_key not in CITIES:
                continue

            for day_offset in range(days_ahead):
                target_date = today + timedelta(days=day_offset)
                month_name = month_names[target_date.month]
                day_num = target_date.day

                # Format: highest-temperature-in-chicago-on-january-24
                slug = f"highest-temperature-in-{city_slug}-on-{month_name}-{day_num}"
                slugs.append((slug, city_key, target_date))

        return slugs

    async def _fetch_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Fetch a single event by its slug."""
        try:
            url = f"{GAMMA_EVENTS_URL}/slug/{slug}"

            async with self._session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    # Market doesn't exist for this date/city
                    return None
                else:
                    logger.debug(f"Event fetch failed: {slug} status={response.status}")
                    return None

        except aiohttp.ClientError as e:
            logger.debug(f"Event fetch error: {slug} error={e}")
            return None

    async def _fetch_gamma_events(self) -> list[dict[str, Any]]:
        """Fetch temperature events from Gamma API using /events/slug/{slug}."""
        all_events: list[dict[str, Any]] = []

        # Generate slugs for cities and upcoming dates
        slugs = self._generate_event_slugs(days_ahead=7)
        logger.info(f"Checking {len(slugs)} potential temperature market slugs")

        # Fetch events in batches to avoid overwhelming the API
        batch_size = 10
        events_found = 0

        for i in range(0, len(slugs), batch_size):
            batch = slugs[i:i + batch_size]
            tasks = [self._fetch_event_by_slug(slug) for slug, _, _ in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (slug, city_key, target_date), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.debug(f"Fetch error for {slug}: {result}")
                    continue

                if result is None:
                    continue

                if not isinstance(result, dict):
                    continue

                events_found += 1

                # Enrich event data with parsed info
                result["_city_key"] = city_key
                result["_target_date"] = target_date.isoformat()
                result["_slug"] = slug

                all_events.append(result)

        total_markets = sum(len(e.get("markets", [])) for e in all_events)
        logger.info(f"Found {events_found} active events with {total_markets} markets")

        return all_events

    def _parse_event_to_market(self, event: dict[str, Any]) -> WeatherMarket | None:
        """
        Parse an event into a WeatherMarket with temperature buckets.

        Each event contains multiple sub-markets, one per temperature range.
        Each sub-market has outcomes ["Yes", "No"] with prices.
        """
        try:
            city_key = event.get("_city_key", "")
            target_date_str = event.get("_target_date", "")
            slug = event.get("_slug", "")

            if not city_key or city_key not in CITIES:
                return None

            city = CITIES[city_key]

            # Parse target date
            target_date_parsed = None
            if target_date_str:
                try:
                    target_date_parsed = date.fromisoformat(target_date_str)
                except ValueError:
                    pass

            # Parse end date
            end_date = None
            end_str = event.get("endDate", "")
            if end_str and isinstance(end_str, str):
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Create the market object
            market = WeatherMarket(
                condition_id=slug,  # Use slug as ID since event doesn't have one
                question_id=event.get("id", ""),
                question=event.get("title", ""),
                description=event.get("description", ""),
                market_slug=slug,
                city=city,
                city_key=city_key,
                target_date=target_date_parsed,
                end_date=end_date,
                volume=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                is_resolved=event.get("closed", False),
                resolution_source=event.get("resolutionSource", ""),
            )

            # Parse each sub-market into a temperature bucket
            sub_markets = event.get("markets", [])
            for sub_market in sub_markets:
                if not isinstance(sub_market, dict):
                    continue

                bucket = self._parse_sub_market_to_bucket(sub_market, city)
                if bucket:
                    market.buckets.append(bucket)
                    # Accumulate volume/liquidity
                    market.volume += float(sub_market.get("volume", 0) or 0)
                    market.liquidity += float(sub_market.get("liquidity", 0) or 0)

            # Sort buckets by temperature
            market.buckets.sort(key=lambda b: (
                b.low_bound if b.low_bound is not None else float('-inf'),
                b.high_bound if b.high_bound is not None else float('inf')
            ))

            return market

        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

    def _parse_sub_market_to_bucket(
        self,
        sub_market: dict[str, Any],
        city: CityConfig,
    ) -> TemperatureBucket | None:
        """
        Parse a sub-market into a TemperatureBucket.

        Each sub-market represents one temperature range (e.g., "5°F or lower").
        The temperature range is extracted from the market's question.
        """
        try:
            question = sub_market.get("question", "")
            if not question:
                return None

            # Extract the temperature range from the question
            # Examples:
            # "Will the highest temperature in Chicago be 5°F or lower?"
            # "Will the highest temperature in Chicago be between 6-7°F?"
            # "Will the highest temperature in Chicago be 16°F or higher?"
            outcome_str = self._extract_temp_range_from_question(question, city.unit)
            if not outcome_str:
                return None

            # Get the YES price from outcomePrices
            outcome_prices = sub_market.get("outcomePrices", "")
            yes_price = 0.0
            if outcome_prices:
                if isinstance(outcome_prices, str):
                    try:
                        prices = json.loads(outcome_prices)
                        if isinstance(prices, list) and len(prices) > 0:
                            yes_price = float(prices[0])
                    except (json.JSONDecodeError, ValueError, IndexError):
                        pass
                elif isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                    yes_price = float(outcome_prices[0])

            # Parse token IDs - clobTokenIds may be a JSON array string
            # Format: [YES_token_id, NO_token_id]
            token_id = ""
            no_token_id = ""
            clob_token_ids = sub_market.get("clobTokenIds", "")
            if clob_token_ids:
                if isinstance(clob_token_ids, str):
                    # Try to parse as JSON array
                    if clob_token_ids.startswith("["):
                        try:
                            ids = json.loads(clob_token_ids)
                            if isinstance(ids, list) and len(ids) > 0:
                                token_id = str(ids[0])  # First token (YES token)
                            if isinstance(ids, list) and len(ids) > 1:
                                no_token_id = str(ids[1])  # Second token (NO token)
                        except json.JSONDecodeError:
                            token_id = clob_token_ids
                    else:
                        token_id = clob_token_ids
                elif isinstance(clob_token_ids, list) and len(clob_token_ids) > 0:
                    token_id = str(clob_token_ids[0])
                    if len(clob_token_ids) > 1:
                        no_token_id = str(clob_token_ids[1])

            # Fallback to conditionId if no token_id
            if not token_id:
                token_id = sub_market.get("conditionId", "")

            bucket = TemperatureBucket(
                token_id=token_id,
                outcome_id=sub_market.get("conditionId", ""),
                outcome=outcome_str,
                yes_price=yes_price,
                no_price=1.0 - yes_price,
                volume_24h=float(sub_market.get("volume24hr", 0) or 0),
                no_token_id=no_token_id,
            )

            # Parse the temperature bounds
            self._parse_bucket_bounds(bucket, outcome_str, city.unit)

            # Debug logging for NO token extraction
            if no_token_id:
                logger.debug(
                    "Extracted NO token",
                    outcome=outcome_str,
                    yes_token=token_id[:20] + "..." if len(token_id) > 20 else token_id,
                    no_token=no_token_id[:20] + "..." if len(no_token_id) > 20 else no_token_id,
                )

            return bucket

        except Exception as e:
            logger.debug(f"Failed to parse sub-market: {e}")
            return None

    def _extract_temp_range_from_question(self, question: str, default_unit: str) -> str | None:
        """Extract temperature range string from a market question."""
        q_lower = question.lower()

        # Pattern: "be X°F or lower" / "be X°C or lower"
        lower_match = re.search(r"be\s+(\d+)\s*°?\s*([FC])?\s+or\s+lower", question, re.IGNORECASE)
        if lower_match:
            temp = lower_match.group(1)
            unit = lower_match.group(2) or default_unit
            return f"<{temp}°{unit.upper()}"

        # Pattern: "be X°F or higher" / "be X°C or higher"
        higher_match = re.search(r"be\s+(\d+)\s*°?\s*([FC])?\s+or\s+higher", question, re.IGNORECASE)
        if higher_match:
            temp = higher_match.group(1)
            unit = higher_match.group(2) or default_unit
            return f"≥{temp}°{unit.upper()}"

        # Pattern: "between X-Y°F" or "be X-Y°F"
        range_match = re.search(r"(?:between|be)\s+(\d+)\s*[-–]\s*(\d+)\s*°?\s*([FC])?", question, re.IGNORECASE)
        if range_match:
            low = range_match.group(1)
            high = range_match.group(2)
            unit = range_match.group(3) or default_unit
            return f"{low}-{high}°{unit.upper()}"

        # Fallback: try to find any temperature pattern
        temp_match = re.search(r"(\d+)\s*°\s*([FC])", question, re.IGNORECASE)
        if temp_match:
            temp = temp_match.group(1)
            unit = temp_match.group(2)
            if "lower" in q_lower or "below" in q_lower:
                return f"<{temp}°{unit.upper()}"
            elif "higher" in q_lower or "above" in q_lower:
                return f"≥{temp}°{unit.upper()}"
            else:
                return f"{temp}°{unit.upper()}"

        return None

    def _parse_market(self, data: dict[str, Any]) -> WeatherMarket | None:
        """Parse a market from Gamma API response."""
        try:
            # Ensure data is a dict before parsing
            if not isinstance(data, dict):
                logger.debug(f"Skipping non-dict market data: {type(data).__name__}")
                return None

            condition_id = data.get("conditionId", data.get("condition_id", ""))
            if not condition_id:
                return None

            question = data.get("question", "")
            description = data.get("description", "")

            # Check for enriched event data (from /events/slug API)
            event_city_key = data.get("_event_city_key", "")
            event_target_date_str = data.get("_event_target_date", "")
            event_description = data.get("_event_description", "")
            event_resolution_source = data.get("_event_resolution_source", "")
            event_end_date = data.get("_event_end_date", "")

            # Use event-level description if market description is empty
            if not description and event_description:
                description = event_description

            # Parse end date (prefer event-level, then market-level)
            end_date = None
            end_str = event_end_date or data.get("endDate") or data.get("end_date_iso")
            if end_str:
                try:
                    if isinstance(end_str, str):
                        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Create market object
            market = WeatherMarket(
                condition_id=condition_id,
                question_id=data.get("questionId", data.get("question_id", "")),
                question=question,
                description=description,
                market_slug=data.get("slug", data.get("market_slug", "")),
                end_date=end_date,
                volume=float(data.get("volume", 0) or 0),
                volume_24h=float(data.get("volume24hr", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                is_resolved=data.get("resolved", False),
                resolution_source=event_resolution_source or data.get("resolutionSource", ""),
            )

            # Parse city - prefer enriched event data
            if event_city_key and event_city_key in CITIES:
                market.city = CITIES[event_city_key]
                market.city_key = event_city_key
            else:
                market.city, market.city_key = self._parse_city(question + " " + description)

            # Parse target date - prefer enriched event data
            if event_target_date_str:
                try:
                    market.target_date = date.fromisoformat(event_target_date_str)
                except ValueError:
                    market.target_date = self._parse_date(question + " " + description)
            else:
                market.target_date = self._parse_date(question + " " + description)

            # Parse temperature buckets from outcomes (primary) or tokens (fallback)
            # The /events/slug API returns outcomes, not tokens
            # Note: outcomes may be a JSON string that needs parsing
            outcomes = data.get("outcomes", data.get("tokens", []))
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except json.JSONDecodeError:
                    outcomes = []
            market.buckets = self._parse_buckets(outcomes, market.city)

            return market

        except Exception as e:
            logger.warning(f"Failed to parse market: {e}")
            return None

    def _is_temperature_market(self, market: WeatherMarket) -> bool:
        """Check if this is a highest temperature market."""
        # Must have a mapped city
        if market.city is None:
            return False

        # Must have at least one bucket
        if len(market.buckets) == 0:
            return False

        # If we have question text, validate it's about temperature
        text = (market.question + " " + market.description).lower()
        if text.strip():
            # Must mention temperature or degrees
            if not any(kw in text for kw in ["temperature", "temp", "degrees", "°", "°f", "°c"]):
                return False

        return True

    def _parse_city(self, text: str) -> tuple[CityConfig | None, str]:
        """Extract city from market text."""
        text_lower = text.lower()

        for city_key, patterns in self.CITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    city_config = CITIES.get(city_key)
                    if city_config:
                        return city_config, city_key

        return None, ""

    def _parse_date(self, text: str) -> date | None:
        """Extract target date from market text."""
        month_map = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }

        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groupdict()

                # Get month
                month_str = groups.get("month", "").lower()
                if month_str.isdigit():
                    month = int(month_str)
                else:
                    month = month_map.get(month_str, 0)

                if not month:
                    continue

                # Get day
                day = int(groups.get("day", 0))
                if not day:
                    continue

                # Get year (default to current/next year)
                year_str = groups.get("year")
                if year_str:
                    year = int(year_str)
                    if year < 100:
                        year += 2000
                else:
                    year = datetime.now().year
                    # If date is in the past this year, assume next year
                    try:
                        parsed = date(year, month, day)
                        if parsed < date.today():
                            year += 1
                    except ValueError:
                        pass

                try:
                    return date(year, month, day)
                except ValueError:
                    continue

        return None

    def _parse_buckets(
        self,
        tokens: list,
        city: CityConfig | None,
    ) -> list[TemperatureBucket]:
        """Parse temperature buckets from token/outcome data."""
        buckets: list[TemperatureBucket] = []
        default_unit = city.unit if city else "F"

        for i, token in enumerate(tokens):
            # Handle both dict format (full market data) and string format (outcome names)
            if isinstance(token, str):
                # Outcome is just a string like "5°F or lower"
                outcome = token
                token_id = ""
                outcome_id = ""
                price = 0.0
                volume = 0.0
            elif isinstance(token, dict):
                outcome = token.get("outcome", token.get("value", ""))
                if not outcome:
                    continue
                token_id = token.get("token_id", token.get("tokenId", ""))
                outcome_id = token.get("outcome_id", token.get("outcomeId", ""))
                price = float(token.get("price", 0) or 0)
                volume = float(token.get("volume", 0) or 0)
            else:
                continue

            if not outcome:
                continue

            bucket = TemperatureBucket(
                token_id=token_id,
                outcome_id=outcome_id,
                outcome=outcome,
                yes_price=price,
                volume_24h=volume,
            )

            # Parse temperature bounds
            self._parse_bucket_bounds(bucket, outcome, default_unit)

            # Calculate NO price
            bucket.no_price = 1.0 - bucket.yes_price

            buckets.append(bucket)

        # Sort by low bound
        buckets.sort(key=lambda b: (
            b.low_bound if b.low_bound is not None else float('-inf'),
            b.high_bound if b.high_bound is not None else float('inf')
        ))

        return buckets

    def _parse_bucket_bounds(
        self,
        bucket: TemperatureBucket,
        outcome: str,
        default_unit: str,
    ) -> None:
        """Parse temperature bounds from outcome string."""
        # Try range pattern first: "68-69°F"
        for pattern in self.BUCKET_PATTERNS:
            match = pattern.search(outcome)
            if not match:
                continue

            groups = match.groupdict()

            # Range pattern
            if "low" in groups and "high" in groups:
                bucket.low_bound = float(groups["low"])
                bucket.high_bound = float(groups["high"])
                # If high < low, assume it's like "68-9" meaning "68-69"
                if bucket.high_bound < bucket.low_bound:
                    bucket.high_bound = bucket.low_bound + (bucket.high_bound % 10)
                bucket.unit = groups.get("unit", default_unit) or default_unit
                return

            # Threshold pattern (>=, <, etc.)
            if "threshold" in groups:
                threshold = float(groups["threshold"])
                bucket.unit = groups.get("unit", default_unit) or default_unit

                # Determine if upper or lower bound
                if any(s in outcome.lower() for s in ["≥", ">=", ">", "over", "+", "higher", "more"]):
                    bucket.low_bound = threshold
                    bucket.high_bound = None  # Open-ended upper
                elif any(s in outcome.lower() for s in ["≤", "<=", "<", "under", "below", "less"]):
                    bucket.low_bound = None  # Open-ended lower
                    bucket.high_bound = threshold
                return

        # Fallback: try to extract any number
        numbers = re.findall(r"(\d+(?:\.\d+)?)", outcome)
        if len(numbers) >= 2:
            bucket.low_bound = float(numbers[0])
            bucket.high_bound = float(numbers[1])
            if bucket.high_bound < bucket.low_bound:
                bucket.high_bound = bucket.low_bound + (bucket.high_bound % 10)
        elif len(numbers) == 1:
            val = float(numbers[0])
            bucket.low_bound = val
            bucket.high_bound = val + 1  # Assume 1-degree bucket

        # Detect unit from outcome string
        if "°c" in outcome.lower() or "celsius" in outcome.lower():
            bucket.unit = "C"
        elif "°f" in outcome.lower() or "fahrenheit" in outcome.lower():
            bucket.unit = "F"
        else:
            bucket.unit = default_unit

    async def _fetch_orderbooks(self, markets: list[WeatherMarket]) -> None:
        """Fetch orderbook data for all market buckets."""
        logger.info(f"Fetching orderbooks for {len(markets)} markets")

        # Collect all token IDs
        token_ids: list[tuple[WeatherMarket, TemperatureBucket]] = []
        for market in markets:
            for bucket in market.buckets:
                if bucket.token_id:
                    token_ids.append((market, bucket))

        # Fetch in batches
        batch_size = 10
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            tasks = [
                self._fetch_orderbook(bucket.token_id)
                for _, bucket in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (market, bucket), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.debug(f"Failed to fetch orderbook for {bucket.token_id}: {result}")
                    continue

                if result:
                    self._apply_orderbook_data(bucket, result)

    async def _fetch_orderbook(self, token_id: str) -> dict[str, Any] | None:
        """Fetch orderbook for a single token."""
        try:
            url = f"{CLOB_API_URL}/book"
            params = {"token_id": token_id}

            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    return None
                return await response.json()

        except Exception:
            return None

    def _apply_orderbook_data(
        self,
        bucket: TemperatureBucket,
        orderbook: dict[str, Any],
    ) -> None:
        """Apply orderbook data to a bucket."""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if bids:
            # Best bid is highest price
            best_bid = max(bids, key=lambda x: float(x.get("price", 0)))
            bucket.best_bid = float(best_bid.get("price", 0))
            bucket.bid_size = float(best_bid.get("size", 0))

        if asks:
            # Best ask is lowest price
            best_ask = min(asks, key=lambda x: float(x.get("price", 0)))
            bucket.best_ask = float(best_ask.get("price", 0))
            bucket.ask_size = float(best_ask.get("size", 0))

        # Calculate spread
        if bucket.best_bid > 0 and bucket.best_ask > 0:
            bucket.spread = bucket.best_ask - bucket.best_bid

        # Update yes_price to midpoint if we have orderbook data
        if bucket.best_bid > 0 and bucket.best_ask > 0:
            bucket.yes_price = bucket.midpoint


def create_mock_markets() -> list[WeatherMarket]:
    """Create mock markets for testing without network access."""
    from datetime import timedelta

    markets = []
    today = date.today()
    tomorrow = today + timedelta(days=1)

    # Mock NYC market - TODAY (for METAR nowcasting demo)
    nyc_market = WeatherMarket(
        condition_id=f"mock_nyc_{today.isoformat()}",
        question_id="mock_q_nyc",
        question=f"What will be the highest temperature in New York City on {today.strftime('%B %d, %Y')}?",
        description="Resolves based on Weather Underground data from KNYC station.",
        market_slug=f"nyc-temp-{today.isoformat()}",
        city=CITIES["nyc"],
        city_key="nyc",
        target_date=today,  # TODAY - METAR nowcasting active
        end_date=datetime.combine(today, datetime.max.time()),
        volume=15000,
        liquidity=5000,
    )

    # Create temperature buckets
    bucket_ranges = [
        ("<32", None, 32),
        ("32-36", 32, 36),
        ("36-40", 36, 40),
        ("40-44", 40, 44),
        ("44-48", 44, 48),
        ("48-52", 48, 52),
        ("≥52", 52, None),
    ]

    prices = [0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]

    for i, ((outcome, low, high), price) in enumerate(zip(bucket_ranges, prices)):
        bucket = TemperatureBucket(
            token_id=f"mock_token_nyc_{i}",
            outcome_id=f"mock_outcome_{i}",
            outcome=f"{outcome}°F",
            low_bound=low,
            high_bound=high,
            unit="F",
            yes_price=price,
            no_price=1 - price,
            best_bid=price - 0.01,
            best_ask=price + 0.01,
            bid_size=500.0,  # $500 liquidity on bid
            ask_size=500.0,  # $500 liquidity on ask
            spread=0.02,
        )
        nyc_market.buckets.append(bucket)

    markets.append(nyc_market)

    # Mock Dallas market - TOMORROW (no METAR, just forecast)
    dallas_market = WeatherMarket(
        condition_id=f"mock_dallas_{tomorrow.isoformat()}",
        question_id="mock_q_dallas",
        question=f"What will be the highest temperature in Dallas on {tomorrow.strftime('%B %d, %Y')}?",
        description="Resolves based on Weather Underground data from KDAL station.",
        market_slug=f"dallas-temp-{tomorrow.isoformat()}",
        city=CITIES["dallas"],
        city_key="dallas",
        target_date=tomorrow,  # TOMORROW - no METAR constraint
        end_date=datetime.combine(tomorrow, datetime.max.time()),
        volume=8000,
        liquidity=3000,
    )

    dallas_ranges = [
        ("<50", None, 50),
        ("50-55", 50, 55),
        ("55-60", 55, 60),
        ("60-65", 60, 65),
        ("65-70", 65, 70),
        ("≥70", 70, None),
    ]

    dallas_prices = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]

    for i, ((outcome, low, high), price) in enumerate(zip(dallas_ranges, dallas_prices)):
        bucket = TemperatureBucket(
            token_id=f"mock_token_dallas_{i}",
            outcome_id=f"mock_outcome_dallas_{i}",
            outcome=f"{outcome}°F",
            low_bound=low,
            high_bound=high,
            unit="F",
            yes_price=price,
            no_price=1 - price,
            best_bid=price - 0.01,
            best_ask=price + 0.01,
            bid_size=500.0,  # $500 liquidity on bid
            ask_size=500.0,  # $500 liquidity on ask
            spread=0.02,
        )
        dallas_market.buckets.append(bucket)

    markets.append(dallas_market)

    return markets


async def test_market_scanner(use_mock: bool = False) -> list[WeatherMarket]:
    """Test the market scanner."""
    print(f"\n{'='*60}")
    print("Polymarket Weather Market Scanner Test")
    print(f"{'='*60}")

    if use_mock:
        print("Mode: MOCK DATA (simulated markets)")
        markets = create_mock_markets()
    else:
        print("Mode: LIVE API")
        async with MarketScanner() as scanner:
            markets = await scanner.scan_weather_markets()

    print(f"{'='*60}\n")
    print(f"Found {len(markets)} weather markets:\n")

    for market in markets:
        print(f"Market: {market.condition_id[:20]}...")
        print(f"  Question: {market.question[:60]}...")
        print(f"  City: {market.city.name if market.city else 'Unknown'}")
        print(f"  Target Date: {market.target_date}")
        print(f"  Hours Until Close: {market.hours_until_close:.1f}")
        print(f"  Buckets: {market.n_buckets}")
        print(f"  Total YES Prices: ${market.total_yes_prices:.3f}")

        if abs(market.arbitrage_gap) > 0.01:
            print(f"  ** ARBITRAGE GAP: {market.arbitrage_gap:.1%} **")

        print(f"\n  Temperature Buckets:")
        for bucket in market.buckets:
            bounds = ""
            if bucket.low_bound is not None and bucket.high_bound is not None:
                bounds = f"[{bucket.low_bound}-{bucket.high_bound})"
            elif bucket.low_bound is not None:
                bounds = f"[{bucket.low_bound}, ∞)"
            elif bucket.high_bound is not None:
                bounds = f"(-∞, {bucket.high_bound})"

            print(f"    {bucket.outcome:12} {bounds:15} YES=${bucket.yes_price:.3f} "
                  f"bid=${bucket.best_bid:.3f} ask=${bucket.best_ask:.3f}")

        print()

    print(f"{'='*60}")
    print("Scan completed!")
    print(f"{'='*60}\n")

    return markets


# Allow running directly for testing
if __name__ == "__main__":
    import sys
    use_mock = "--mock" in sys.argv
    try:
        asyncio.run(test_market_scanner(use_mock=use_mock))
    except Exception as e:
        if "aiohttp" in str(type(e).__module__) or "ClientError" in str(type(e)):
            print("\nNetwork unavailable, falling back to mock data...\n")
            asyncio.run(test_market_scanner(use_mock=True))
        else:
            raise
