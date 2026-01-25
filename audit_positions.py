#!/usr/bin/env python3
"""Audit open positions against current forecasts."""

import asyncio
import re
from datetime import datetime, timedelta

import aiohttp

from src.config import CITIES, CityConfig, get_settings
from src.data.ensemble import OpenMeteoClient, EnsembleForecast


async def fetch_positions(session: aiohttp.ClientSession, funder: str) -> list[dict]:
    """Fetch positions from Polymarket data API."""
    url = f"https://data-api.polymarket.com/positions?user={funder}"
    async with session.get(url, timeout=30) as response:
        if response.status != 200:
            print(f"Failed to fetch positions: {response.status}")
            return []
        return await response.json()


async def fetch_market_info(session: aiohttp.ClientSession, token_id: str) -> dict | None:
    """Fetch market info for a token."""
    url = f"https://gamma-api.polymarket.com/markets?clob_token_ids={token_id}"
    try:
        async with session.get(url, timeout=30) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data[0] if data else None
    except Exception as e:
        print(f"Error fetching market {token_id}: {e}")
        return None


def parse_market_info(market: dict) -> dict | None:
    """Parse market info to extract city, date, and bucket."""
    question = market.get("question", "")

    # City patterns
    city_patterns = {
        "new york": "nyc",
        "nyc": "nyc",
        "london": "london",
        "seoul": "seoul",
        "dallas": "dallas",
        "toronto": "toronto",
        "seattle": "seattle",
        "atlanta": "atlanta",
        "chicago": "chicago",
        "miami": "miami",
        "ankara": "ankara",
        "buenos aires": "buenos_aires",
        "wellington": "wellington",
    }

    city_key = None
    for pattern, key in city_patterns.items():
        if pattern.lower() in question.lower():
            city_key = key
            break

    # Date patterns - look for "on January 25" or similar
    date_match = re.search(r"on\s+(\w+)\s+(\d{1,2})", question)
    target_date = None
    if date_match:
        month_str, day = date_match.groups()
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        month = month_map.get(month_str.lower())
        if month:
            year = 2026 if month >= 1 else 2025  # Assume current year context
            target_date = f"{year}-{month:02d}-{int(day):02d}"

    # Get the outcome (bucket)
    outcome = market.get("outcome", "")

    return {
        "city_key": city_key,
        "target_date": target_date,
        "outcome": outcome,
        "question": question,
        "condition_id": market.get("condition_id"),
    }


def parse_bucket_bounds(outcome: str, unit: str) -> tuple[float | None, float | None]:
    """Parse bucket outcome to get low/high bounds."""
    outcome = outcome.strip()

    # Handle "X or below" / "≤X"
    below_match = re.match(r"[<≤]?\s*(\d+)[°]?\s*[°]?[FC]?\s*(or below)?", outcome, re.IGNORECASE)
    if below_match or "or below" in outcome.lower():
        temp_match = re.search(r"(\d+)", outcome)
        if temp_match:
            return None, float(temp_match.group(1))

    # Handle "X or above" / "≥X"
    above_match = re.match(r"[>≥]?\s*(\d+)[°]?\s*[°]?[FC]?\s*(or above)?", outcome, re.IGNORECASE)
    if above_match or "or above" in outcome.lower():
        temp_match = re.search(r"(\d+)", outcome)
        if temp_match:
            return float(temp_match.group(1)), None

    # Handle "X-Y°F" range
    range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", outcome)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    return None, None


async def get_forecast(city_key: str, target_date_str: str) -> EnsembleForecast | None:
    """Get ensemble forecast for city and date."""
    if city_key not in CITIES:
        return None

    city = CITIES[city_key]

    # Parse date string to date object
    from datetime import date
    try:
        parts = target_date_str.split("-")
        target_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        print(f"Could not parse date: {target_date_str}")
        return None

    client = OpenMeteoClient()

    try:
        forecast = await client.get_ensemble_forecast(
            city=city,
            target_date=target_date
        )
        return forecast
    except Exception as e:
        print(f"Error getting forecast for {city_key}: {e}")
        return None
    finally:
        await client.close()


async def audit_positions(funder_override: str | None = None):
    """Main audit function."""
    settings = get_settings()
    funder = funder_override or settings.polymarket_funder

    if not funder:
        print("ERROR: No POLYMARKET_FUNDER configured")
        print("Usage: python audit_positions.py <wallet_address>")
        return

    print(f"Auditing positions for: {funder}\n")
    print("=" * 80)

    async with aiohttp.ClientSession() as session:
        # Fetch all positions
        positions = await fetch_positions(session, funder)

        if not positions:
            print("No positions found")
            return

        # Filter to weather markets with actual size
        weather_positions = []
        for pos in positions:
            size = float(pos.get("size", 0))
            if size <= 0:
                continue

            token_id = pos.get("asset")
            market = await fetch_market_info(session, token_id)

            if not market:
                continue

            question = market.get("question", "").lower()
            if "temperature" not in question and "°f" not in question and "°c" not in question:
                continue

            parsed = parse_market_info(market)
            if parsed and parsed["city_key"]:
                weather_positions.append({
                    **pos,
                    **parsed,
                    "market": market,
                })

        print(f"Found {len(weather_positions)} weather positions\n")

        # Group by city/date for forecast efficiency
        forecasts = {}

        for pos in weather_positions:
            city_key = pos["city_key"]
            target_date = pos["target_date"]
            outcome = pos["outcome"]
            size = float(pos.get("size", 0))
            avg_price = float(pos.get("avgPrice", 0))
            cur_price = float(pos.get("curPrice", pos.get("price", 0)))
            current_value = float(pos.get("currentValue", 0))
            cashout_value = float(pos.get("cashoutValue", 0))

            # Skip resolved positions
            if current_value == 0 and size > 0:
                continue

            print(f"\n{'─' * 80}")
            print(f"📍 {CITIES[city_key].name} - {target_date}")
            print(f"   Bucket: {outcome}")
            print(f"   Size: {size:.1f} shares @ {avg_price*100:.1f}¢ avg")
            print(f"   Current Price: {cur_price*100:.1f}¢")
            print(f"   Value: ${current_value:.2f} (cashout: ${cashout_value:.2f})")

            # Get forecast
            cache_key = f"{city_key}_{target_date}"
            if cache_key not in forecasts:
                print(f"   Fetching forecast...")
                forecasts[cache_key] = await get_forecast(city_key, target_date)

            forecast = forecasts.get(cache_key)

            if forecast and forecast.high_temps is not None and len(forecast.high_temps) > 0:
                temps = forecast.high_temps  # numpy array
                import numpy as np
                mean_temp = float(np.mean(temps))
                min_temp = float(np.min(temps))
                max_temp = float(np.max(temps))

                print(f"   Forecast: mean={mean_temp:.1f}° min={min_temp:.1f}° max={max_temp:.1f}° ({len(temps)} members)")

                # Parse bucket bounds
                city = CITIES[city_key]
                low, high = parse_bucket_bounds(outcome, city.unit)

                if low is not None or high is not None:
                    # Calculate probability from ensemble
                    count_in_bucket = 0
                    for t in temps:
                        if low is None:  # "X or below"
                            if t <= high:
                                count_in_bucket += 1
                        elif high is None:  # "X or above"
                            if t >= low:
                                count_in_bucket += 1
                        else:  # Range
                            if low <= t < high:
                                count_in_bucket += 1

                    fair_prob = count_in_bucket / len(temps)

                    # Calculate edge
                    if cur_price > 0:
                        edge = (fair_prob - cur_price) / cur_price
                    else:
                        edge = 0

                    print(f"   Fair Value: {fair_prob*100:.1f}% ({count_in_bucket}/{len(temps)} members)")
                    print(f"   Edge vs Current: {edge*100:+.1f}%")

                    # Entry edge
                    if avg_price > 0:
                        entry_edge = (fair_prob - avg_price) / avg_price
                        print(f"   Edge vs Entry: {entry_edge*100:+.1f}%")

                    # Assessment
                    print()
                    if fair_prob < 0.02 and cur_price < 0.05:
                        print(f"   ⚠️  ASSESSMENT: Long-shot lottery ticket")
                    elif edge > 0.10:
                        print(f"   ✅ ASSESSMENT: Still has positive edge ({edge*100:.0f}%)")
                    elif edge < -0.20:
                        print(f"   🔴 ASSESSMENT: Significant negative edge - consider exit")
                    elif edge < -0.10:
                        print(f"   🟡 ASSESSMENT: Edge has compressed - monitor closely")
                    else:
                        print(f"   🟢 ASSESSMENT: Position looks reasonable")
                else:
                    print(f"   ⚠️  Could not parse bucket bounds: {outcome}")
            else:
                print(f"   ⚠️  Could not fetch forecast")

    print(f"\n{'=' * 80}")
    print("Audit complete")


if __name__ == "__main__":
    import sys
    funder_addr = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(audit_positions(funder_addr))
