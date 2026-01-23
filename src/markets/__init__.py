"""Polymarket integration module."""

from src.markets.scanner import MarketScanner, WeatherMarket, TemperatureBucket
from src.markets.client import PolymarketClient
from src.markets.market_scanner import (
    MarketScanner as WeatherMarketScanner,
    WeatherMarket as WeatherMarketV2,
    TemperatureBucket as TemperatureBucketV2,
    create_mock_markets,
)

__all__ = [
    "MarketScanner",
    "WeatherMarket",
    "TemperatureBucket",
    "PolymarketClient",
    "WeatherMarketScanner",
    "WeatherMarketV2",
    "TemperatureBucketV2",
    "create_mock_markets",
]
