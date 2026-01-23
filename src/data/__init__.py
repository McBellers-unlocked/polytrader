"""Weather data aggregation module."""

from src.data.ensemble import EnsembleForecast, OpenMeteoClient
from src.data.metar import MetarClient, MetarObservation
from src.data.tomorrow import TomorrowClient, TomorrowForecast
from src.data.aggregator import WeatherAggregator, AggregatedForecast
from src.data.open_meteo import (
    OpenMeteoClient as OpenMeteoEnsembleClient,
    EnsembleForecastResult,
    EnsembleMember,
)
from src.data.nowcasting import (
    MetarNowcaster,
    TemperatureConstraint,
    create_mock_constraint,
)

__all__ = [
    "EnsembleForecast",
    "OpenMeteoClient",
    "OpenMeteoEnsembleClient",
    "EnsembleForecastResult",
    "EnsembleMember",
    "MetarClient",
    "MetarObservation",
    "TomorrowClient",
    "TomorrowForecast",
    "WeatherAggregator",
    "AggregatedForecast",
    # Nowcasting
    "MetarNowcaster",
    "TemperatureConstraint",
    "create_mock_constraint",
]
