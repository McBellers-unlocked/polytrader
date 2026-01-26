#!/usr/bin/env python3
"""
Trade Analysis Script

Analyzes existing positions by fetching ensemble forecasts and calculating
fair values to evaluate trade quality and expected outcomes.
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

import numpy as np

from src.config import CITIES, CityConfig, get_settings
from src.data.aggregator import WeatherAggregator


@dataclass
class Trade:
    """Represents a single trade position."""
    city: str
    target_date: date
    bucket: str  # e.g., "66-67°F" or "43°F or below"
    side: str  # "YES" or "NO"
    entry_price: float  # Price paid (e.g., 0.80 for 80¢)
    current_price: float
    shares: float
    cost_basis: float
    current_value: float
    pnl: float
    pnl_pct: float

    # Parsed bucket bounds
    low_bound: Optional[float] = None
    high_bound: Optional[float] = None
    is_above: bool = False  # "X or higher"
    is_below: bool = False  # "X or below"

    @property
    def city_key(self) -> str:
        """Get the city key for config lookup."""
        mapping = {
            "Miami": "miami",
            "Seattle": "seattle",
            "London": "london",
            "Atlanta": "atlanta",
            "New York": "nyc",
            "NYC": "nyc",
            "Dallas": "dallas",
            "Chicago": "chicago",
            "Toronto": "toronto",
        }
        return mapping.get(self.city, self.city.lower())


def parse_bucket(bucket: str, unit: str) -> tuple[Optional[float], Optional[float], bool, bool]:
    """
    Parse a bucket string into bounds.

    Returns: (low_bound, high_bound, is_above, is_below)
    """
    bucket = bucket.replace("°F", "").replace("°C", "").strip()

    # Handle "X or higher" / "X or above"
    if "or higher" in bucket or "or above" in bucket:
        temp = float(bucket.split()[0])
        return (temp, None, True, False)

    # Handle "X or below" / "X or lower"
    if "or below" in bucket or "or lower" in bucket:
        temp = float(bucket.split()[0])
        return (None, temp + 1, False, True)  # +1 because it's "X or below" = less than X+1

    # Handle range "X-Y"
    if "-" in bucket:
        parts = bucket.split("-")
        low = float(parts[0])
        high = float(parts[1]) + 1  # Exclusive upper bound
        return (low, high, False, False)

    # Single temperature
    try:
        temp = float(bucket)
        return (temp, temp + 1, False, False)
    except ValueError:
        return (None, None, False, False)


# ============================================================================
# TRADE DATA FROM SCREENSHOTS
# ============================================================================

# Parse your trades from the screenshots
# Note: Dates inferred from visible text - London shows "January 27"
# Miami/Seattle/Atlanta dates appear to be Jan 26-27 based on context

TRADES = [
    # ===================
    # LONDON - January 27
    # ===================
    Trade(
        city="London", target_date=date(2026, 1, 27),
        bucket="7°C", side="YES",
        entry_price=0.01, current_price=0.01,
        shares=351.5, cost_basis=4.57, current_value=351.53,
        pnl=0.18, pnl_pct=3.85
    ),
    Trade(
        city="London", target_date=date(2026, 1, 27),
        bucket="6°C", side="YES",
        entry_price=0.009, current_price=0.008,
        shares=182.2, cost_basis=1.64, current_value=182.22,
        pnl=-0.27, pnl_pct=-16.67
    ),
    Trade(
        city="London", target_date=date(2026, 1, 27),
        bucket="8°C", side="YES",
        entry_price=0.14, current_price=0.11,
        shares=33.8, cost_basis=4.77, current_value=33.82,
        pnl=-1.05, pnl_pct=-21.93
    ),
    Trade(
        city="London", target_date=date(2026, 1, 27),
        bucket="9°C or higher", side="NO",
        entry_price=0.19, current_price=0.13,
        shares=18.0, cost_basis=3.42, current_value=18.00,
        pnl=-1.17, pnl_pct=-34.21
    ),

    # ===================
    # SEATTLE - January 27 (inferred)
    # ===================
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="48-49°F", side="NO",
        entry_price=0.75, current_price=0.59,
        shares=5.0, cost_basis=3.75, current_value=5.00,
        pnl=-0.80, pnl_pct=-21.33
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="43°F or below", side="YES",
        entry_price=0.01, current_price=0.03,
        shares=293.0, cost_basis=2.93, current_value=293.00,
        pnl=4.98, pnl_pct=170.0
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="44-45°F", side="YES",
        entry_price=0.06, current_price=0.03,
        shares=83.8, cost_basis=5.03, current_value=83.77,
        pnl=-2.51, pnl_pct=-50.0
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="46-47°F", side="NO",
        entry_price=0.72, current_price=0.50,
        shares=9.5, cost_basis=6.85, current_value=9.51,
        pnl=-2.14, pnl_pct=-31.25
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="44-45°F", side="NO",
        entry_price=0.74, current_price=0.71,
        shares=9.3, cost_basis=6.85, current_value=9.25,
        pnl=-0.28, pnl_pct=-4.05
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="41°F or below", side="NO",
        entry_price=0.83, current_price=0.88,
        shares=8.3, cost_basis=6.85, current_value=8.25,
        pnl=0.41, pnl_pct=6.02
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="48-49°F", side="NO",
        entry_price=0.90, current_price=0.79,
        shares=7.6, cost_basis=6.85, current_value=7.61,
        pnl=-0.81, pnl_pct=-11.83
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="50-51°F", side="NO",
        entry_price=0.75, current_price=0.50,
        shares=5.0, cost_basis=3.75, current_value=5.00,
        pnl=-1.25, pnl_pct=-33.33
    ),
    Trade(
        city="Seattle", target_date=date(2026, 1, 27),
        bucket="46-47°F", side="YES",
        entry_price=0.12, current_price=0.11,
        shares=5.0, cost_basis=0.60, current_value=5.00,
        pnl=-0.08, pnl_pct=-12.5
    ),

    # ===================
    # ATLANTA - January 27 (inferred)
    # ===================
    Trade(
        city="Atlanta", target_date=date(2026, 1, 27),
        bucket="33°F or below", side="YES",
        entry_price=0.02, current_price=0.05,
        shares=171.5, cost_basis=3.43, current_value=171.49,
        pnl=5.06, pnl_pct=147.5
    ),
    Trade(
        city="Atlanta", target_date=date(2026, 1, 27),
        bucket="40-41°F", side="YES",
        entry_price=0.05, current_price=0.03,
        shares=97.2, cost_basis=4.57, current_value=97.23,
        pnl=-2.14, pnl_pct=-46.81
    ),
    Trade(
        city="Atlanta", target_date=date(2026, 1, 27),
        bucket="36-37°F", side="YES",
        entry_price=0.10, current_price=0.23,
        shares=34.3, cost_basis=3.43, current_value=34.29,
        pnl=4.41, pnl_pct=128.73
    ),
    Trade(
        city="Atlanta", target_date=date(2026, 1, 27),
        bucket="42-43°F", side="NO",
        entry_price=0.88, current_price=0.94,
        shares=5.0, cost_basis=4.40, current_value=5.00,
        pnl=0.28, pnl_pct=6.25
    ),
    Trade(
        city="Atlanta", target_date=date(2026, 1, 27),
        bucket="40-41°F", side="NO",
        entry_price=0.75, current_price=0.64,
        shares=5.0, cost_basis=3.75, current_value=5.00,
        pnl=-0.58, pnl_pct=-15.33
    ),

    # ===================
    # MIAMI - January 26 (inferred from "on J..." text)
    # ===================
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="66-67°F", side="NO",
        entry_price=0.80, current_price=0.63,
        shares=5.0, cost_basis=4.00, current_value=5.00,
        pnl=-0.85, pnl_pct=-21.24
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="68-69°F", side="NO",
        entry_price=0.90, current_price=0.50,
        shares=5.0, cost_basis=4.50, current_value=5.00,
        pnl=-2.00, pnl_pct=-44.44
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="70°F or higher", side="YES",
        entry_price=0.28, current_price=0.17,
        shares=25.2, cost_basis=7.13, current_value=25.15,
        pnl=-2.98, pnl_pct=-41.76
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="82-83°F", side="NO",
        entry_price=0.82, current_price=0.85,
        shares=8.4, cost_basis=6.85, current_value=8.35,
        pnl=0.25, pnl_pct=3.66
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="86-87°F", side="NO",
        entry_price=0.90, current_price=0.73,
        shares=7.6, cost_basis=6.85, current_value=7.61,
        pnl=-1.29, pnl_pct=-18.89
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="64-65°F", side="NO",
        entry_price=0.87, current_price=0.86,
        shares=5.3, cost_basis=4.57, current_value=5.25,
        pnl=-0.08, pnl_pct=-1.71
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="84-85°F", side="NO",
        entry_price=0.66, current_price=0.57,
        shares=5.2, cost_basis=3.43, current_value=5.19,
        pnl=-0.47, pnl_pct=-13.65
    ),
    Trade(
        city="Miami", target_date=date(2026, 1, 26),
        bucket="80-81°F", side="NO",
        entry_price=0.89, current_price=0.97,
        shares=5.0, cost_basis=4.45, current_value=5.00,
        pnl=0.38, pnl_pct=8.43
    ),
]


def group_trades_by_city_date(trades: list[Trade]) -> dict[tuple[str, date], list[Trade]]:
    """Group trades by (city, target_date)."""
    groups: dict[tuple[str, date], list[Trade]] = {}
    for trade in trades:
        key = (trade.city, trade.target_date)
        if key not in groups:
            groups[key] = []
        groups[key].append(trade)
    return groups


async def analyze_trades():
    """Main analysis function."""
    print("=" * 80)
    print("POLYTRADER TRADE ANALYSIS")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Parse bucket bounds for all trades
    for trade in TRADES:
        city_config = CITIES.get(trade.city_key)
        unit = city_config.unit if city_config else "F"
        low, high, is_above, is_below = parse_bucket(trade.bucket, unit)
        trade.low_bound = low
        trade.high_bound = high
        trade.is_above = is_above
        trade.is_below = is_below

    # Group trades
    groups = group_trades_by_city_date(TRADES)

    # Summary stats
    total_cost = sum(t.cost_basis for t in TRADES)
    total_pnl = sum(t.pnl for t in TRADES)

    print(f"\nTotal Trades: {len(TRADES)}")
    print(f"Total Cost Basis: ${total_cost:.2f}")
    print(f"Total P&L: ${total_pnl:+.2f} ({100*total_pnl/total_cost:+.1f}%)")
    print(f"\nMarkets: {len(groups)}")

    # Analyze each group
    async with WeatherAggregator() as aggregator:
        for (city, target_date), trades in sorted(groups.items()):
            print("\n" + "=" * 80)
            print(f"📍 {city.upper()} - {target_date.strftime('%B %d, %Y')}")
            print("=" * 80)

            city_key = trades[0].city_key
            city_config = CITIES.get(city_key)

            if not city_config:
                print(f"⚠️  City '{city}' not found in config")
                continue

            # Check if date is in the future
            today = datetime.now().date()
            if target_date < today:
                print(f"⚠️  Market has SETTLED (target date was {target_date})")
                days_ago = (today - target_date).days
                print(f"   Resolution was {days_ago} day(s) ago")
            elif target_date == today:
                print("🔴 Market SETTLES TODAY")
            else:
                days_until = (target_date - today).days
                print(f"📅 Settles in {days_until} day(s)")

            # Fetch ensemble forecast (if date is in future)
            if target_date >= today:
                try:
                    forecast = await aggregator.get_forecast(city_config, target_date)
                    stats = forecast.get_distribution_stats()

                    print(f"\n🌡️  ENSEMBLE FORECAST ({forecast.ensemble.n_members if forecast.ensemble else 0} members)")
                    print(f"   Mean High: {stats.get('mean', 0):.1f}°{city_config.unit}")
                    print(f"   Median:    {stats.get('median', 0):.1f}°{city_config.unit}")
                    print(f"   Std Dev:   {stats.get('std', 0):.1f}°")
                    print(f"   Range:     {stats.get('min', 0):.1f}° - {stats.get('max', 0):.1f}°")
                    print(f"   P10-P90:   {stats.get('p10', 0):.1f}° - {stats.get('p90', 0):.1f}°")
                    print(f"   Model Agreement: {forecast.model_agreement:.1%}")
                    print(f"   Confidence: {forecast.confidence:.1%}")

                except Exception as e:
                    print(f"\n⚠️  Could not fetch forecast: {e}")
                    forecast = None
            else:
                forecast = None

            # Analyze each trade
            print(f"\n📊 POSITIONS ({len(trades)} trades)")
            print("-" * 75)

            group_cost = 0
            group_pnl = 0

            for trade in sorted(trades, key=lambda t: abs(t.pnl), reverse=True):
                group_cost += trade.cost_basis
                group_pnl += trade.pnl

                # Calculate fair value if we have forecast
                fair_value = None
                edge = None

                if forecast and forecast.ensemble:
                    if trade.is_above and trade.low_bound is not None:
                        fair_value = forecast.probability_above(trade.low_bound)
                    elif trade.is_below and trade.high_bound is not None:
                        fair_value = forecast.probability_below(trade.high_bound)
                    elif trade.low_bound is not None and trade.high_bound is not None:
                        fair_value = forecast.probability_in_range(trade.low_bound, trade.high_bound)

                    if fair_value is not None:
                        # For NO positions, fair value of NO = 1 - fair value of YES
                        if trade.side == "NO":
                            no_fair_value = 1 - fair_value
                            edge = (no_fair_value - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0
                        else:
                            edge = (fair_value - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0

                # Format output
                pnl_color = "🟢" if trade.pnl >= 0 else "🔴"

                print(f"\n{pnl_color} {trade.side:3} {trade.bucket:15} @ {trade.entry_price:.2f} → {trade.current_price:.2f}")
                print(f"   Shares: {trade.shares:.1f} | Cost: ${trade.cost_basis:.2f} | P&L: ${trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)")

                if fair_value is not None:
                    yes_fv = fair_value
                    no_fv = 1 - fair_value

                    if trade.side == "YES":
                        print(f"   Fair Value (YES): {yes_fv:.1%} | Entry Edge: {edge:+.1%}" if edge else f"   Fair Value (YES): {yes_fv:.1%}")
                        if yes_fv > trade.entry_price:
                            print(f"   ✅ UNDERPRICED at entry - good buy")
                        elif yes_fv < trade.entry_price * 0.8:
                            print(f"   ⚠️  OVERPRICED at entry - edge was negative")
                    else:
                        print(f"   Fair Value (NO): {no_fv:.1%} | Entry Edge: {edge:+.1%}" if edge else f"   Fair Value (NO): {no_fv:.1%}")
                        if no_fv > trade.entry_price:
                            print(f"   ✅ UNDERPRICED at entry - good buy")
                        elif no_fv < trade.entry_price * 0.8:
                            print(f"   ⚠️  OVERPRICED at entry - edge was negative")

            print("\n" + "-" * 75)
            pnl_pct = 100 * group_pnl / group_cost if group_cost > 0 else 0
            summary_icon = "🟢" if group_pnl >= 0 else "🔴"
            print(f"{summary_icon} {city} TOTAL: Cost ${group_cost:.2f} | P&L ${group_pnl:+.2f} ({pnl_pct:+.1f}%)")

    # Final summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    # By city
    print("\nBy City:")
    city_stats: dict[str, dict] = {}
    for trade in TRADES:
        if trade.city not in city_stats:
            city_stats[trade.city] = {"cost": 0, "pnl": 0, "trades": 0}
        city_stats[trade.city]["cost"] += trade.cost_basis
        city_stats[trade.city]["pnl"] += trade.pnl
        city_stats[trade.city]["trades"] += 1

    for city, stats in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        pct = 100 * stats["pnl"] / stats["cost"] if stats["cost"] > 0 else 0
        icon = "🟢" if stats["pnl"] >= 0 else "🔴"
        print(f"  {icon} {city:10} ({stats['trades']:2} trades): ${stats['pnl']:+7.2f} ({pct:+6.1f}%)")

    # Winners vs Losers
    winners = [t for t in TRADES if t.pnl >= 0]
    losers = [t for t in TRADES if t.pnl < 0]

    print(f"\nWin Rate: {len(winners)}/{len(TRADES)} ({100*len(winners)/len(TRADES):.1f}%)")
    print(f"Total Wins:   ${sum(t.pnl for t in winners):+.2f}")
    print(f"Total Losses: ${sum(t.pnl for t in losers):+.2f}")

    print(f"\n{'='*80}")
    print(f"NET P&L: ${total_pnl:+.2f} ({100*total_pnl/total_cost:+.1f}%)")
    print(f"{'='*80}")


def analyze_without_api():
    """
    Analyze trades without requiring API access.
    Provides insights based on trade data and position characteristics.
    """
    print("=" * 80)
    print("POLYTRADER TRADE ANALYSIS (Static Mode)")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Parse bucket bounds for all trades
    for trade in TRADES:
        city_config = CITIES.get(trade.city_key)
        unit = city_config.unit if city_config else "F"
        low, high, is_above, is_below = parse_bucket(trade.bucket, unit)
        trade.low_bound = low
        trade.high_bound = high
        trade.is_above = is_above
        trade.is_below = is_below

    # Summary stats
    total_cost = sum(t.cost_basis for t in TRADES)
    total_pnl = sum(t.pnl for t in TRADES)

    print(f"\nTotal Trades: {len(TRADES)}")
    print(f"Total Cost Basis: ${total_cost:.2f}")
    print(f"Total P&L: ${total_pnl:+.2f} ({100*total_pnl/total_cost:+.1f}%)")

    # Group by city
    groups = group_trades_by_city_date(TRADES)

    # Analyze each city/date combination
    for (city, target_date), trades in sorted(groups.items()):
        print("\n" + "=" * 80)
        print(f"📍 {city.upper()} - {target_date.strftime('%B %d, %Y')}")
        print("=" * 80)

        city_key = trades[0].city_key
        city_config = CITIES.get(city_key)

        if city_config:
            unit = city_config.unit
            print(f"   Unit: °{unit} | Timezone: {city_config.timezone}")

        # Market timing
        today = datetime.now().date()
        if target_date < today:
            print(f"   ⚠️  SETTLED (target date passed)")
        elif target_date == today:
            print("   🔴 SETTLES TODAY")
        else:
            print(f"   📅 Settles in {(target_date - today).days} day(s)")

        # Analyze position distribution
        yes_positions = [t for t in trades if t.side == "YES"]
        no_positions = [t for t in trades if t.side == "NO"]

        print(f"\n📊 Position Summary:")
        print(f"   YES positions: {len(yes_positions)} (${sum(t.cost_basis for t in yes_positions):.2f})")
        print(f"   NO positions:  {len(no_positions)} (${sum(t.cost_basis for t in no_positions):.2f})")

        # Temperature range analysis
        all_bounds = []
        for t in trades:
            if t.low_bound is not None:
                all_bounds.append(t.low_bound)
            if t.high_bound is not None:
                all_bounds.append(t.high_bound)

        if all_bounds:
            print(f"   Temp range covered: {min(all_bounds):.0f}° - {max(all_bounds):.0f}°{unit if city_config else ''}")

        # Identify position strategy
        print("\n📈 Position Analysis:")
        tail_bets = []
        center_bets = []

        for t in trades:
            if t.is_below or t.is_above:
                tail_bets.append(t)
            elif t.low_bound and t.high_bound:
                center_bets.append(t)

        if tail_bets:
            print(f"   Tail bets (extreme temps): {len(tail_bets)}")
            for t in tail_bets:
                icon = "🟢" if t.pnl >= 0 else "🔴"
                print(f"     {icon} {t.side} {t.bucket}: ${t.pnl:+.2f} ({t.pnl_pct:+.1f}%)")

        if center_bets:
            print(f"   Range bets: {len(center_bets)}")

        # Individual positions
        print(f"\n📋 All Positions ({len(trades)} trades):")
        print("-" * 70)

        group_cost = 0
        group_pnl = 0

        for trade in sorted(trades, key=lambda t: (t.bucket, t.side)):
            group_cost += trade.cost_basis
            group_pnl += trade.pnl

            icon = "🟢" if trade.pnl >= 0 else "🔴"
            price_move = "↑" if trade.current_price > trade.entry_price else "↓" if trade.current_price < trade.entry_price else "="

            print(f"  {icon} {trade.side:3} {trade.bucket:15} {trade.entry_price:.2f}{price_move}{trade.current_price:.2f} "
                  f"| ${trade.cost_basis:.2f} → ${trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)")

        print("-" * 70)
        pnl_pct = 100 * group_pnl / group_cost if group_cost > 0 else 0
        summary_icon = "🟢" if group_pnl >= 0 else "🔴"
        print(f"{summary_icon} {city} TOTAL: ${group_cost:.2f} invested → ${group_pnl:+.2f} ({pnl_pct:+.1f}%)")

    # =========================================================================
    # STRATEGIC ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGIC ANALYSIS")
    print("=" * 80)

    # By strategy type
    print("\n🎯 By Strategy Type:")

    # YES vs NO
    yes_trades = [t for t in TRADES if t.side == "YES"]
    no_trades = [t for t in TRADES if t.side == "NO"]

    yes_pnl = sum(t.pnl for t in yes_trades)
    no_pnl = sum(t.pnl for t in no_trades)
    yes_cost = sum(t.cost_basis for t in yes_trades)
    no_cost = sum(t.cost_basis for t in no_trades)

    print(f"   YES positions ({len(yes_trades)}): ${yes_pnl:+.2f} ({100*yes_pnl/yes_cost:+.1f}%)")
    print(f"   NO positions ({len(no_trades)}): ${no_pnl:+.2f} ({100*no_pnl/no_cost:+.1f}%)")

    # Tail vs center
    tail_trades = [t for t in TRADES if t.is_below or t.is_above]
    center_trades = [t for t in TRADES if not t.is_below and not t.is_above]

    if tail_trades:
        tail_pnl = sum(t.pnl for t in tail_trades)
        tail_cost = sum(t.cost_basis for t in tail_trades)
        print(f"   Tail bets ({len(tail_trades)}): ${tail_pnl:+.2f} ({100*tail_pnl/tail_cost:+.1f}%)")

    if center_trades:
        center_pnl = sum(t.pnl for t in center_trades)
        center_cost = sum(t.cost_basis for t in center_trades)
        print(f"   Center bets ({len(center_trades)}): ${center_pnl:+.2f} ({100*center_pnl/center_cost:+.1f}%)")

    # Entry price analysis
    print("\n💰 By Entry Price:")
    cheap_trades = [t for t in TRADES if t.entry_price < 0.20]
    mid_trades = [t for t in TRADES if 0.20 <= t.entry_price < 0.70]
    expensive_trades = [t for t in TRADES if t.entry_price >= 0.70]

    for name, trades_subset in [("Cheap (<20¢)", cheap_trades),
                                 ("Mid (20-70¢)", mid_trades),
                                 ("Expensive (≥70¢)", expensive_trades)]:
        if trades_subset:
            pnl = sum(t.pnl for t in trades_subset)
            cost = sum(t.cost_basis for t in trades_subset)
            pct = 100 * pnl / cost if cost > 0 else 0
            icon = "🟢" if pnl >= 0 else "🔴"
            print(f"   {icon} {name} ({len(trades_subset)}): ${pnl:+.2f} ({pct:+.1f}%)")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS & RECOMMENDATIONS")
    print("=" * 80)

    print("""
📊 OBSERVATIONS:

1. WIN RATE: 30.8% is below break-even for most position types
   - Need higher accuracy or better position sizing

2. ATLANTA (+35.9%): Best performer - cold snap trades working
   - 33°F or below YES: +147.5% (big winner)
   - 36-37°F YES: +128.7% (big winner)
   - Strategy: Cold side bets paid off

3. MIAMI (-16.9%): Worst performer
   - 70°F or higher YES: -41.8% (biggest loser)
   - Multiple NO positions on extreme temps losing
   - Issue: Appears to have bet on bimodal distribution
     but temps settled in the middle

4. SEATTLE (-5.7%): Mixed results
   - 43°F or below YES: +170% (big winner)
   - But multiple NO positions on mid-40s losing
   - Conflicting positions (both YES and NO on similar buckets)

5. LONDON (-16.0%): All YES positions on 6-8°C range
   - Relatively tight grouping suggests conviction on range
   - But NO on 9°C+ also losing suggests range was wrong

🔧 REFINEMENT SUGGESTIONS:

1. AVOID CONFLICTING POSITIONS
   - Seattle has YES 44-45°F and NO 44-45°F simultaneously
   - This creates drag on returns

2. POSITION SIZING
   - Cheap YES bets (1-10¢) have asymmetric payoff
   - But need higher hit rate or better timing

3. TAIL RISK BETS
   - Working well for Atlanta cold snap
   - Be cautious with extreme temp bets in Miami
     (climate rarely hits 66°F highs in winter)

4. MODEL AGREEMENT CHECK
   - Ensure >65% model agreement before entry
   - Check ensemble spread for confidence

5. ENTRY TIMING
   - Some positions entered at extremes (90¢ NO)
   - Consider waiting for better pricing
""")

    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # By city
    print("\nBy City Performance:")
    city_stats: dict[str, dict] = {}
    for trade in TRADES:
        if trade.city not in city_stats:
            city_stats[trade.city] = {"cost": 0, "pnl": 0, "trades": 0}
        city_stats[trade.city]["cost"] += trade.cost_basis
        city_stats[trade.city]["pnl"] += trade.pnl
        city_stats[trade.city]["trades"] += 1

    for city, stats in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        pct = 100 * stats["pnl"] / stats["cost"] if stats["cost"] > 0 else 0
        icon = "🟢" if stats["pnl"] >= 0 else "🔴"
        print(f"  {icon} {city:10} ({stats['trades']:2} trades): ${stats['pnl']:+7.2f} ({pct:+6.1f}%)")

    print(f"\n{'='*80}")
    print(f"NET P&L: ${total_pnl:+.2f} ({100*total_pnl/total_cost:+.1f}%)")
    print(f"Win Rate: {len([t for t in TRADES if t.pnl >= 0])}/{len(TRADES)} ({100*len([t for t in TRADES if t.pnl >= 0])/len(TRADES):.1f}%)")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Try with API first, fall back to static analysis
    try:
        asyncio.run(analyze_trades())
    except Exception as e:
        print(f"Note: API unavailable ({e}), running static analysis...")
        analyze_without_api()
