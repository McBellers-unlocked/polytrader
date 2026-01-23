"""Tests for Open-Meteo ensemble client."""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.config import CityConfig, CITIES
from src.data.ensemble import OpenMeteoClient, EnsembleForecast


@pytest.fixture
def city_config():
    """Create a test city config."""
    return CITIES["nyc"]


@pytest.fixture
def sample_forecast(city_config):
    """Create a sample forecast."""
    return EnsembleForecast(
        city=city_config,
        target_date=date.today(),
        fetch_time=datetime.utcnow(),
        temperatures=np.random.normal(70, 5, (24, 50)),
        high_temps=np.random.normal(75, 3, 50),
        low_temps=np.random.normal(60, 3, 50),
        n_members=50,
        model_names=["gfs_seamless", "ecmwf_ifs04"],
    )


class TestEnsembleForecast:
    """Tests for EnsembleForecast class."""

    def test_probability_in_range(self, sample_forecast):
        """Test probability calculation for temperature range."""
        # Should return a value between 0 and 1
        prob = sample_forecast.probability_in_range(70, 80)
        assert 0 <= prob <= 1

    def test_probability_above(self, sample_forecast):
        """Test probability above threshold."""
        prob = sample_forecast.probability_above(70)
        assert 0 <= prob <= 1

    def test_probability_below(self, sample_forecast):
        """Test probability below threshold."""
        prob = sample_forecast.probability_below(80)
        assert 0 <= prob <= 1

    def test_distribution_stats(self, sample_forecast):
        """Test distribution statistics."""
        stats = sample_forecast.get_distribution_stats()

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p10" in stats
        assert "p90" in stats

        # Mean should be around 75 for our sample
        assert 65 < stats["mean"] < 85

    def test_get_percentile(self, sample_forecast):
        """Test percentile calculation."""
        p50 = sample_forecast.get_percentile(50)
        p10 = sample_forecast.get_percentile(10)
        p90 = sample_forecast.get_percentile(90)

        assert p10 < p50 < p90


class TestOpenMeteoClient:
    """Tests for OpenMeteoClient class."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        client = OpenMeteoClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit(self):
        """Test temperature conversion."""
        temps = np.array([0, 10, 20, 30, 100])
        converted = OpenMeteoClient._celsius_to_fahrenheit(temps)

        assert converted[0] == 32  # 0°C = 32°F
        assert abs(converted[2] - 68) < 0.1  # 20°C ≈ 68°F
        assert converted[4] == 212  # 100°C = 212°F
