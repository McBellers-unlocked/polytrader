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
        # Test 1: Try EVENTS endpoint (temperature markets are events!)
        print(f"\n[TEST 1] Fetch from EVENTS endpoint")
        print(f"URL: {GAMMA_EVENTS_URL}")

        params = {"active": "true", "closed": "false", "limit": 100}
        async with session.get(GAMMA_EVENTS_URL, params=params, timeout=30) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response type: {type(data).__name__}, Length: {len(data) if isinstance(data, list) else 'N/A'}")

            if isinstance(data, list):
                temp_events = []
                for item in data:
                    if isinstance(item, dict):
                        title = item.get("title", "") or item.get("question", "")
                        slug = item.get("slug", "")
                        if "temperature" in title.lower() or "temperature" in slug.lower():
                            temp_events.append({
                                "title": title[:60],
                                "slug": slug,
                                "markets": len(item.get("markets", [])),
                            })

                print(f"\nFound {len(temp_events)} temperature events:")
                for e in temp_events[:10]:
                    print(f"  - {e['title']}...")
                    print(f"    slug: {e['slug']}")
                    print(f"    markets: {e['markets']}")

        # Test 2: Fetch specific event by slug
        print(f"\n[TEST 2] Fetch specific London temperature event")
        slug = "highest-temperature-in-london-on-january-23"
        url = f"{GAMMA_EVENTS_URL}/{slug}"
        print(f"URL: {url}")

        async with session.get(url, timeout=30) as response:
            print(f"Status: {response.status}")
            if response.status == 200:
                data = await response.json()
                print(f"Response type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    print(f"Title: {data.get('title', 'N/A')}")
                    markets = data.get("markets", [])
                    print(f"Markets count: {len(markets)}")
                    if markets and isinstance(markets[0], dict):
                        print(f"First market keys: {list(markets[0].keys())[:10]}")
                        print(f"First market question: {markets[0].get('question', 'N/A')[:60]}")
                        # Check for tokens
                        tokens = markets[0].get("tokens", markets[0].get("outcomes", []))
                        print(f"Tokens/outcomes: {len(tokens)}")

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
