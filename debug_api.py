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
        # Fetch with weather tag
        params = {
            "active": "true",
            "closed": "false",
            "tag": "weather",
            "limit": 10,  # Just get 10 for debugging
        }

        print(f"\nFetching from: {GAMMA_API_URL}")
        print(f"Params: {params}")

        async with session.get(GAMMA_API_URL, params=params, timeout=30) as response:
            print(f"\nStatus: {response.status}")
            print(f"Content-Type: {response.headers.get('content-type')}")

            data = await response.json()

            print(f"\nResponse type: {type(data).__name__}")

            if isinstance(data, list):
                print(f"List length: {len(data)}")
                print("\nFirst 5 items:")
                for i, item in enumerate(data[:5]):
                    print(f"\n  [{i}] Type: {type(item).__name__}")
                    if isinstance(item, dict):
                        print(f"      Keys: {list(item.keys())[:10]}...")
                        if "question" in item:
                            print(f"      Question: {item.get('question', '')[:60]}...")
                        if "conditionId" in item:
                            print(f"      conditionId: {item.get('conditionId', '')[:30]}...")
                    elif isinstance(item, str):
                        print(f"      Value: {item[:100]}...")
                    else:
                        print(f"      Value: {str(item)[:100]}...")
            elif isinstance(data, dict):
                print(f"Dict keys: {list(data.keys())}")
                # Check if markets are nested
                for key in ["markets", "data", "results"]:
                    if key in data:
                        print(f"\nFound '{key}' key with {len(data[key])} items")
                        if data[key]:
                            print(f"First item type: {type(data[key][0]).__name__}")
            else:
                print(f"Unexpected response type: {type(data)}")
                print(f"Raw: {str(data)[:500]}")

    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(debug_api())
