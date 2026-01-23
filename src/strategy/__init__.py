"""Strategy and signal generation module."""

from src.strategy.fair_value import FairValueCalculator, BucketProbability
from src.strategy.edge import EdgeDetector, TradeSignal
from src.strategy.arbitrage import ArbitrageDetector, ArbitrageOpportunity
from src.strategy.probability import (
    FairValueCalculator as KDEFairValueCalculator,
    FairValueResult,
    MarketFairValue,
    EdgeDetector as KDEEdgeDetector,
    ArbitrageDetector as KDEArbitrageDetector,
    ArbitrageOpportunity as KDEArbitrageOpportunity,
)

__all__ = [
    "FairValueCalculator",
    "BucketProbability",
    "EdgeDetector",
    "TradeSignal",
    "ArbitrageDetector",
    "ArbitrageOpportunity",
    "KDEFairValueCalculator",
    "FairValueResult",
    "MarketFairValue",
    "KDEEdgeDetector",
    "KDEArbitrageDetector",
    "KDEArbitrageOpportunity",
]
