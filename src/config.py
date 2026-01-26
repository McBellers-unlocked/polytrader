"""Configuration management using pydantic-settings."""

from enum import Enum
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading mode enumeration."""

    PAPER = "paper"  # No real trades, just logging
    SEMI = "semi"  # Requires manual confirmation
    AUTO = "auto"  # Fully automated


class CityConfig:
    """Configuration for a tracked city."""

    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        metar: str,
        unit: str,
        timezone: str = "UTC",
    ):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.metar = metar
        self.unit = unit  # "F" or "C"
        self.timezone = timezone

    def __repr__(self) -> str:
        return f"CityConfig({self.name}, {self.lat}, {self.lon}, {self.metar}, {self.unit})"


# Predefined city configurations
CITIES: dict[str, CityConfig] = {
    "nyc": CityConfig(
        name="New York City",
        lat=40.71,
        lon=-74.01,
        metar="KNYC",
        unit="F",
        timezone="America/New_York",
    ),
    "london": CityConfig(
        name="London",
        lat=51.51,
        lon=-0.13,
        metar="EGLL",
        unit="C",
        timezone="Europe/London",
    ),
    "seoul": CityConfig(
        name="Seoul",
        lat=37.57,
        lon=126.98,
        metar="RKSI",
        unit="C",
        timezone="Asia/Seoul",
    ),
    "dallas": CityConfig(
        name="Dallas",
        lat=32.78,
        lon=-96.80,
        metar="KDAL",
        unit="F",
        timezone="America/Chicago",
    ),
    "toronto": CityConfig(
        name="Toronto",
        lat=43.65,
        lon=-79.38,
        metar="CYYZ",
        unit="C",
        timezone="America/Toronto",
    ),
    "seattle": CityConfig(
        name="Seattle",
        lat=47.61,
        lon=-122.33,
        metar="KSEA",
        unit="F",
        timezone="America/Los_Angeles",
    ),
    "atlanta": CityConfig(
        name="Atlanta",
        lat=33.75,
        lon=-84.39,
        metar="KATL",
        unit="F",
        timezone="America/New_York",
    ),
    "chicago": CityConfig(
        name="Chicago",
        lat=41.88,
        lon=-87.63,
        metar="KORD",
        unit="F",
        timezone="America/Chicago",
    ),
    "miami": CityConfig(
        name="Miami",
        lat=25.76,
        lon=-80.19,
        metar="KMIA",
        unit="F",
        timezone="America/New_York",
    ),
    "ankara": CityConfig(
        name="Ankara",
        lat=39.93,
        lon=32.86,
        metar="LTAC",
        unit="C",
        timezone="Europe/Istanbul",
    ),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Trading mode
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)

    # Bankroll
    starting_bankroll: float = Field(default=1500.0, ge=0)

    # Polymarket credentials
    polymarket_private_key: str = Field(default="")
    polymarket_funder: str = Field(default="")
    polymarket_api_key: str = Field(default="")

    # API keys
    tomorrow_io_api_key: str = Field(default="")

    # Risk parameters
    max_position_pct: float = Field(default=0.02, ge=0, le=1)
    daily_loss_stop_pct: float = Field(default=0.05, ge=0, le=1)
    max_drawdown_pct: float = Field(default=0.20, ge=0, le=1)

    # Edge thresholds
    min_edge_threshold: float = Field(default=0.15, ge=0, le=1)  # Increased from 0.10
    min_model_agreement: float = Field(default=0.65, ge=0, le=1)
    min_confidence: float = Field(default=0.50, ge=0, le=1)  # New: minimum confidence

    # Liquidity requirements
    max_spread_pct: float = Field(default=0.10, ge=0, le=1)  # Max 10% bid-ask spread
    min_liquidity_usd: float = Field(default=100.0, ge=0)  # Min $100 liquidity

    # Position limits
    max_positions_per_market: int = Field(default=3, ge=1)  # Focus bets
    max_total_positions: int = Field(default=15, ge=1)  # Avoid over-diversification

    # Order execution
    order_timeout_seconds: int = Field(default=60, ge=1)
    max_slippage_pct: float = Field(default=0.02, ge=0, le=1)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Database
    db_path: str = Field(default="polytrader.db")

    # API URLs
    polymarket_clob_url: str = Field(default="https://clob.polymarket.com")
    polymarket_gamma_url: str = Field(default="https://gamma-api.polymarket.com")
    open_meteo_url: str = Field(default="https://ensemble-api.open-meteo.com/v1/ensemble")
    tomorrow_io_url: str = Field(default="https://api.tomorrow.io/v4")
    aviation_weather_url: str = Field(
        default="https://aviationweather.gov/api/data/metar"
    )

    @field_validator("trading_mode", mode="before")
    @classmethod
    def parse_trading_mode(cls, v: Any) -> TradingMode:
        if isinstance(v, TradingMode):
            return v
        if isinstance(v, str):
            return TradingMode(v.lower())
        raise ValueError(f"Invalid trading mode: {v}")

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.trading_mode == TradingMode.PAPER

    @property
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode (semi or auto)."""
        return self.trading_mode in (TradingMode.SEMI, TradingMode.AUTO)

    @property
    def requires_confirmation(self) -> bool:
        """Check if trades require manual confirmation."""
        return self.trading_mode == TradingMode.SEMI

    def validate_live_trading(self) -> None:
        """Validate that credentials are set for live trading."""
        if self.is_live_trading:
            if not self.polymarket_private_key:
                raise ValueError("POLYMARKET_PRIVATE_KEY required for live trading")
            if not self.polymarket_funder:
                raise ValueError("POLYMARKET_FUNDER required for live trading")


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
