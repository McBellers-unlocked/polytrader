#!/usr/bin/env python3
"""Debug script to inspect Polymarket API response."""

import asyncio
import aiohttp
import json

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


async def debug_api():
    print("=" * 60)
    print("Debugging Polymarket Gamma API Response")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Test 0: Try different API parameters to find weather markets
        print(f"\n[TEST 0] Try different API query parameters")

        test_params = [
            {"tag": "weather", "active": "true", "limit": 50},
            {"tag": "climate", "active": "true", "limit": 50},
            {"tag": "climate-science", "active": "true", "limit": 50},
            {"category": "weather", "active": "true", "limit": 50},
            {"topic": "weather", "active": "true", "limit": 50},
            {"slug_contains": "temperature", "active": "true", "limit": 50},
            {"title_contains": "temperature", "active": "true", "limit": 50},
            {"q": "temperature", "active": "true", "limit": 50},
            {"search": "temperature", "active": "true", "limit": 50},
        ]

        for params in test_params:
            async with session.get(GAMMA_EVENTS_URL, params=params, timeout=30) as response:
                data = await response.json()
                count = len(data) if isinstance(data, list) else 0
                temp_count = 0
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            title = (item.get("title", "") or "").lower()
                            if "temperature" in title:
                                temp_count += 1
                print(f"  {params} -> {count} results, {temp_count} temperature")
                if temp_count > 0:
                    print(f"    ^ FOUND TEMPERATURE MARKETS!")
                    # Show first few
                    for item in data[:3]:
                        if isinstance(item, dict) and "temperature" in (item.get("title", "") or "").lower():
                            print(f"      - {item.get('title', '')[:60]}")
                            print(f"        slug: {item.get('slug', '')}")

        # Test 1: Try EVENTS endpoint - search broadly
        print(f"\n[TEST 1] Fetch from EVENTS endpoint")
        print(f"URL: {GAMMA_EVENTS_URL}")

        params = {"active": "true", "closed": "false", "limit": 200}
        async with session.get(GAMMA_EVENTS_URL, params=params, timeout=30) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response type: {type(data).__name__}, Length: {len(data) if isinstance(data, list) else 'N/A'}")

            if isinstance(data, list):
                # Search for weather-related keywords
                keywords = ["temperature", "temp", "highest", "weather", "degrees", "celsius", "fahrenheit"]
                weather_events = []

                print(f"\nSearching for keywords: {keywords}")
                for item in data:
                    if isinstance(item, dict):
                        title = (item.get("title", "") or "").lower()
                        slug = (item.get("slug", "") or "").lower()
                        desc = (item.get("description", "") or "").lower()
                        combined = title + " " + slug + " " + desc

                        if any(kw in combined for kw in keywords):
                            weather_events.append({
                                "title": item.get("title", "")[:60],
                                "slug": item.get("slug", ""),
                                "markets": len(item.get("markets", [])),
                            })

                print(f"\nFound {len(weather_events)} weather-related events:")
                for e in weather_events[:15]:
                    print(f"  - {e['title']}")
                    print(f"    slug: {e['slug']}, markets: {e['markets']}")

                # Also show first 5 event titles to see what's there
                print(f"\nFirst 10 event titles (for reference):")
                for item in data[:10]:
                    if isinstance(item, dict):
                        print(f"  - {item.get('title', 'N/A')[:70]}")

        # Test 2: Try different slug formats for temperature events
        print(f"\n[TEST 2] Try different event slug formats")
        slugs_to_try = [
            "highest-temperature-in-london-on-january-24",  # Tomorrow
            "highest-temperature-in-london-on-january-25",
            "highest-temperature-in-new-york-city-on-january-24",
            "highest-temp-london-january-24",
            "london-temperature-january-24",
        ]

        for slug in slugs_to_try:
            url = f"{GAMMA_EVENTS_URL}/{slug}"
            async with session.get(url, timeout=30) as response:
                status = response.status
                if status == 200:
                    data = await response.json()
                    markets = data.get("markets", [])
                    print(f"  FOUND: {slug} -> {len(markets)} markets")
                    if markets:
                        print(f"         First market: {markets[0].get('question', 'N/A')[:50]}...")
                    break
                else:
                    print(f"  {status}: {slug}")
        else:
            print("  No temperature events found with these slugs")

        # Test 3: Check what the /markets endpoint returns for comparison
        print(f"\n[TEST 3] Check /markets endpoint structure")
        params = {"active": "true", "closed": "false", "limit": 5}
        async with session.get(GAMMA_API_URL, params=params, timeout=30) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict):
                    print(f"Market keys: {list(item.keys())}")
                    print(f"Has 'tokens': {'tokens' in item}")
                    print(f"Has 'outcomes': {'outcomes' in item}")

    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(debug_api())
