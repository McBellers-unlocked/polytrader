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
        # Test 0: Search MARKETS endpoint (not events) with high limit
        print(f"\n[TEST 0] Search /markets endpoint with high limit")

        # Try markets endpoint with very high limit
        for limit in [500, 1000]:
            params = {"active": "true", "closed": "false", "limit": limit}
            async with session.get(GAMMA_API_URL, params=params, timeout=60) as response:
                data = await response.json()
                count = len(data) if isinstance(data, list) else 0
                temp_count = 0
                temp_samples = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            q = (item.get("question", "") or "").lower()
                            if "temperature" in q or "highest temp" in q:
                                temp_count += 1
                                if len(temp_samples) < 5:
                                    temp_samples.append(item)
                print(f"  /markets limit={limit} -> {count} results, {temp_count} temperature markets")
                if temp_samples:
                    print(f"    FOUND TEMPERATURE MARKETS!")
                    for item in temp_samples:
                        print(f"      - {item.get('question', '')[:60]}")
                        print(f"        conditionId: {item.get('conditionId', '')[:30]}...")
                        print(f"        slug: {item.get('slug', '')}")

        # Try events with very high limit
        print(f"\n  Trying /events with high limits...")
        for limit in [500, 1000]:
            params = {"active": "true", "closed": "false", "limit": limit}
            async with session.get(GAMMA_EVENTS_URL, params=params, timeout=60) as response:
                data = await response.json()
                count = len(data) if isinstance(data, list) else 0
                temp_count = 0
                temp_samples = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            title = (item.get("title", "") or "").lower()
                            if "temperature" in title:
                                temp_count += 1
                                if len(temp_samples) < 5:
                                    temp_samples.append(item)
                print(f"  /events limit={limit} -> {count} results, {temp_count} temperature events")
                if temp_samples:
                    print(f"    FOUND TEMPERATURE EVENTS!")
                    for item in temp_samples:
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

        # Test 2: Try the EXACT slug format from working URL
        print(f"\n[TEST 2] Try exact slug formats from polymarket.com")
        slugs_to_try = [
            # From user's URL: https://polymarket.com/event/highest-temperature-in-chicago-on-january-24
            "highest-temperature-in-chicago-on-january-24",
            "highest-temperature-in-london-on-january-24",
            "highest-temperature-in-atlanta-on-january-24",
            "highest-temperature-in-seoul-on-january-24",
            "highest-temperature-in-miami-on-january-24",
            "highest-temperature-in-new-york-city-on-january-24",
            "highest-temperature-in-nyc-on-january-24",
        ]

        found_any = False
        for slug in slugs_to_try:
            url = f"{GAMMA_EVENTS_URL}/{slug}"
            async with session.get(url, timeout=30) as response:
                status = response.status
                print(f"  {status}: {slug}")
                if status == 200:
                    found_any = True
                    data = await response.json()
                    print(f"    Keys: {list(data.keys())[:10]}")
                    markets = data.get("markets", [])
                    print(f"    Markets: {len(markets)}")
                    if markets and isinstance(markets[0], dict):
                        m = markets[0]
                        print(f"    First market question: {m.get('question', 'N/A')[:50]}")
                        print(f"    First market conditionId: {m.get('conditionId', 'N/A')[:30]}")
                        print(f"    Has outcomes: {'outcomes' in m}, Has tokens: {'tokens' in m}")
                        outcomes = m.get("outcomes", m.get("tokens", []))
                        print(f"    Outcomes count: {len(outcomes) if outcomes else 0}")
                        if outcomes:
                            print(f"    First outcome: {outcomes[0] if outcomes else 'N/A'}")

        if not found_any:
            print("  No events found - trying /markets endpoint directly...")
            # Maybe they're accessible via /markets with the slug
            for slug in slugs_to_try[:3]:
                url = f"{GAMMA_API_URL}/{slug}"
                async with session.get(url, timeout=30) as response:
                    print(f"  /markets/{slug}: {response.status}")

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
