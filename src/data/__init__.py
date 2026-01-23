"""Weather data aggregation module."""

from src.data.ensemble import EnsembleForecast, OpenMeteoClient
from src.data.metar import MetarClient, MetarObservation
from src.data.tomorrow import TomorrowClient, TomorrowForecast
from src.data.aggregator import WeatherAggregator, AggregatedForecast

__all__ = [
    "EnsembleForecast",
    "OpenMeteoClient",
    "MetarClient",
    "MetarObservation",
    "TomorrowClient",
    "TomorrowForecast",
    "WeatherAggregator",
    "AggregatedForecast",
]
