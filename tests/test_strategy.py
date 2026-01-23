"""Tests for strategy components."""

import pytest
from decimal import Decimal
from datetime import date, datetime
from unittest.mock import MagicMock
import numpy as np

from src.config import CITIES
from src.data.aggregator import AggregatedForecast
from src.markets.scanner import WeatherMarket, TemperatureBucket
from src.strategy.fair_value import FairValueCalculator, BucketProbability
from src.strategy.edge import EdgeDetector, TradeSignal, SignalType
from src.strategy.arbitrage import ArbitrageDetector


@pytest.fixture
def city_config():
    """Create a test city config."""
    return CITIES["nyc"]


@pytest.fixture
def sample_forecast(city_config):
    """Create a sample aggregated forecast."""
    # Create temps centered around 72°F with 3°F std dev
    temps = np.random.normal(72, 3, 100)
    return AggregatedForecast(
        city=city_config,
        target_date=date.today(),
        fetch_time=datetime.utcnow(),
        combined_high_temps=temps,
        model_agreement=0.75,
        confidence=0.8,
    )


@pytest.fixture
def sample_market(city_config):
    """Create a sample weather market."""
    buckets = [
        TemperatureBucket(
            token_id="token_68_70",
            outcome="68-70°F",
            low_bound=68,
            high_bound=70,
            yes_price=0.15,
            no_price=0.85,
            yes_bid=0.14,
            yes_ask=0.16,
            no_bid=0.84,
            no_ask=0.86,
        ),
        TemperatureBucket(
            token_id="token_70_72",
            outcome="70-72°F",
            low_bound=70,
            high_bound=72,
            yes_price=0.25,
            no_price=0.75,
            yes_bid=0.24,
            yes_ask=0.26,
            no_bid=0.74,
            no_ask=0.76,
        ),
        TemperatureBucket(
            token_id="token_72_74",
            outcome="72-74°F",
            low_bound=72,
            high_bound=74,
            yes_price=0.25,
            no_price=0.75,
            yes_bid=0.24,
            yes_ask=0.26,
            no_bid=0.74,
            no_ask=0.76,
        ),
        TemperatureBucket(
            token_id="token_74_76",
            outcome="74-76°F",
            low_bound=74,
            high_bound=76,
            yes_price=0.20,
            no_price=0.80,
            yes_bid=0.19,
            yes_ask=0.21,
            no_bid=0.79,
            no_ask=0.81,
        ),
    ]

    return WeatherMarket(
        condition_id="test_market_123",
        question="Highest temperature in NYC on Jan 25?",
        description="Weather prediction market",
        city=city_config,
        target_date=date.today(),
        end_date=datetime.utcnow(),
        buckets=buckets,
    )


class TestFairValueCalculator:
    """Tests for FairValueCalculator."""

    def test_calculate_fair_values(self, sample_market, sample_forecast):
        """Test fair value calculation."""
        calc = FairValueCalculator()
        probs = calc.calculate_fair_values(sample_market, sample_forecast)

        assert len(probs) == len(sample_market.buckets)

        # All probabilities should be between 0 and 1
        for bp in probs:
            assert 0 <= bp.fair_value <= 1

        # Total should be approximately 1
        total = sum(bp.fair_value for bp in probs)
        # Note: might not sum to 1 if forecast extends beyond buckets
        assert total > 0

    def test_calibrate_probabilities(self, sample_market, sample_forecast):
        """Test probability calibration."""
        calc = FairValueCalculator()
        probs = calc.calculate_fair_values(sample_market, sample_forecast)
        calibrated = calc.calibrate_probabilities(probs)

        # After calibration, total should be very close to 1
        total = sum(bp.fair_value for bp in calibrated)
        assert abs(total - 1.0) < 0.01

    def test_edge_calculation(self, sample_market, sample_forecast):
        """Test edge calculation."""
        calc = FairValueCalculator()
        probs = calc.calculate_fair_values(sample_market, sample_forecast)

        for bp in probs:
            if bp.market_price > 0:
                expected_edge = (bp.fair_value - bp.market_price) / bp.market_price
                assert abs(bp.edge - expected_edge) < 0.001


class TestEdgeDetector:
    """Tests for EdgeDetector."""

    def test_detect_signals(self, sample_market, sample_forecast):
        """Test signal detection."""
        calc = FairValueCalculator()
        probs = calc.calculate_fair_values(sample_market, sample_forecast)

        detector = EdgeDetector()
        signals = detector.detect_signals(
            sample_market,
            probs,
            model_agreement=0.75,
            bankroll=Decimal("1000"),
        )

        # Should return list of signals
        assert isinstance(signals, list)

        # All signals should have edge above threshold
        for signal in signals:
            assert abs(signal.edge) >= 0.10

    def test_filter_signals(self):
        """Test signal filtering."""
        detector = EdgeDetector()

        # Create mock signals
        signals = [
            MagicMock(edge=0.15, confidence=0.8, expected_value=0.05),
            MagicMock(edge=0.25, confidence=0.7, expected_value=0.08),
            MagicMock(edge=0.10, confidence=0.9, expected_value=0.03),
        ]

        filtered = detector.filter_signals(signals, max_signals=2)

        assert len(filtered) == 2


class TestArbitrageDetector:
    """Tests for ArbitrageDetector."""

    def test_detect_arbitrage_with_gap(self, sample_market):
        """Test arbitrage detection when sum < 1."""
        # Current total is 0.15 + 0.25 + 0.25 + 0.20 = 0.85
        # This should be detected as arbitrage

        detector = ArbitrageDetector()
        arb = detector.detect_arbitrage(sample_market)

        assert arb is not None
        assert arb.total_yes_price < 1.0
        assert arb.profit_margin > 0

    def test_detect_no_arbitrage(self, sample_market, city_config):
        """Test no arbitrage when sum ≈ 1."""
        # Create market with prices summing to ~1
        buckets = [
            TemperatureBucket(
                token_id="token_1",
                outcome="68-72°F",
                low_bound=68,
                high_bound=72,
                yes_price=0.50,
                no_price=0.50,
                yes_bid=0.49,
                yes_ask=0.51,
                no_bid=0.49,
                no_ask=0.51,
            ),
            TemperatureBucket(
                token_id="token_2",
                outcome="72-76°F",
                low_bound=72,
                high_bound=76,
                yes_price=0.50,
                no_price=0.50,
                yes_bid=0.49,
                yes_ask=0.51,
                no_bid=0.49,
                no_ask=0.51,
            ),
        ]

        market = WeatherMarket(
            condition_id="test_market",
            question="Test",
            description="Test",
            city=city_config,
            target_date=date.today(),
            end_date=datetime.utcnow(),
            buckets=buckets,
        )

        detector = ArbitrageDetector()
        arb = detector.detect_arbitrage(market)

        assert arb is None  # No arbitrage opportunity
