#!/usr/bin/env python3
"""Debug script to inspect Polymarket API response."""

import asyncio
import aiohttp
import json

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"


async def debug_api():
    print("=" * 60)
    print("Debugging Polymarket Gamma API Response")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Test 1: Fetch with weather tag
        params = {
            "active": "true",
            "closed": "false",
            "tag": "weather",
            "limit": 10,
        }

        print(f"\n[TEST 1] Fetch with tag='weather'")
        print(f"URL: {GAMMA_API_URL}")
        print(f"Params: {params}")

        async with session.get(GAMMA_API_URL, params=params, timeout=30) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response type: {type(data).__name__}, Length: {len(data) if isinstance(data, list) else 'N/A'}")

            # Count types
            if isinstance(data, list):
                type_counts = {}
                for item in data:
                    t = type(item).__name__
                    type_counts[t] = type_counts.get(t, 0) + 1
                print(f"Item types: {type_counts}")

        # Test 2: Fetch WITHOUT tag (this is where most markets come from)
        params2 = {
            "active": "true",
            "closed": "false",
            "limit": 100,
        }

        print(f"\n[TEST 2] Fetch WITHOUT tag (limit=100)")
        print(f"Params: {params2}")

        async with session.get(GAMMA_API_URL, params=params2, timeout=30) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Response type: {type(data).__name__}, Length: {len(data) if isinstance(data, list) else 'N/A'}")

            if isinstance(data, list):
                type_counts = {}
                str_samples = []
                for item in data:
                    t = type(item).__name__
                    type_counts[t] = type_counts.get(t, 0) + 1
                    if isinstance(item, str) and len(str_samples) < 3:
                        str_samples.append(item[:80])
                print(f"Item types: {type_counts}")
                if str_samples:
                    print(f"String samples: {str_samples}")

        # Test 3: Search for actual temperature markets
        print(f"\n[TEST 3] Search for 'temperature' in questions")
        params3 = {
            "active": "true",
            "closed": "false",
            "limit": 200,
        }

        async with session.get(GAMMA_API_URL, params=params3, timeout=30) as response:
            data = await response.json()
            temp_markets = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        q = item.get("question", "").lower()
                        if "temperature" in q or "temp" in q or "degrees" in q:
                            temp_markets.append(item.get("question", "")[:70])

            print(f"Found {len(temp_markets)} temperature-related markets:")
            for m in temp_markets[:5]:
                print(f"  - {m}...")

    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(debug_api())
