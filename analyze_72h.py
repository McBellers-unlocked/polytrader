"""Deep analysis of weather trading performance over the last 72 hours."""

import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import re


async def analyze_72h():
    """Comprehensive analysis of last 72 hours of weather trading."""
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'
    now = datetime.now(timezone.utc)
    cutoff_72h = now - timedelta(hours=72)

    async with aiohttp.ClientSession() as session:
        # Fetch activity history
        act_url = f'https://data-api.polymarket.com/activity?user={address}&limit=500'
        async with session.get(act_url) as resp:
            activity = await resp.json()

    weather_keywords = ['temperature', 'highest', '°F', '°C']

    # Parse all weather-related activities
    trades = []
    for act in activity:
        title = act.get('title', '') or act.get('description', '') or ''
        if not any(kw.lower() in title.lower() for kw in weather_keywords):
            continue

        # Parse timestamp
        ts_str = act.get('timestamp') or act.get('createdAt') or ''
        try:
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                ts = now  # Default to now if no timestamp
        except:
            ts = now

        # Skip if older than 72h
        if ts < cutoff_72h:
            continue

        # Get activity type - API may use different field names
        act_type = (act.get('type', '') or act.get('action', '') or '').lower()
        outcome = act.get('outcome', '') or act.get('side', '')
        price = float(act.get('price', 0) or act.get('avgPrice', 0) or 0)
        size = float(act.get('size', 0) or act.get('shares', 0) or 0)
        value = float(act.get('value', 0) or act.get('usdcSize', 0) or act.get('amount', 0) or 0)

        # Also check for transaction-based data
        if value == 0 and size > 0 and price > 0:
            value = size * price

        # Extract city from title
        city = extract_city(title)

        # Extract temperature bucket from title
        bucket = extract_bucket(title)

        # Extract market date from title
        market_date = extract_date(title)

        # Calculate hours ago
        hours_ago = (now - ts).total_seconds() / 3600

        trades.append({
            'timestamp': ts,
            'hours_ago': hours_ago,
            'type': act_type,
            'title': title,
            'outcome': outcome,
            'price': price,
            'size': size,
            'value': value,
            'city': city,
            'bucket': bucket,
            'market_date': market_date,
            'raw': act,  # Keep raw data for debugging
        })

    # Sort by timestamp (newest first)
    trades.sort(key=lambda x: x['timestamp'], reverse=True)

    print("=" * 90)
    print("DEEP TRADE ANALYSIS - LAST 72 HOURS")
    print(f"Analysis time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 90)

    # Debug: Show first few raw activities to understand the API format
    if trades:
        print("\n[DEBUG] Sample raw activity data:")
        sample = trades[0]['raw']
        for key in sorted(sample.keys())[:15]:
            print(f"  {key}: {sample.get(key)}")

    # Debug: Print all unique types found
    unique_types = set(t['type'] for t in trades)
    print(f"\nAPI activity types found: {unique_types}")

    # Separate by trade type - be flexible with naming
    buys = [t for t in trades if t['type'] in ['buy', 'bought', 'purchase', 'open']]
    claims = [t for t in trades if t['type'] in ['claim', 'redeem', 'claimed', 'won', 'payout', 'settle']]
    losses = [t for t in trades if t['type'] in ['lost', 'loss', 'expired', 'resolve']]
    sells = [t for t in trades if t['type'] in ['sell', 'sold', 'close']]

    # Catch-all for unknown types
    unknown = [t for t in trades if t not in buys + claims + losses + sells]
    if unknown:
        print(f"Unknown activity types: {set(t['type'] for t in unknown)}")

    print(f"\nTotal activities in 72h: {len(trades)}")
    print(f"  Buys: {len(buys)}")
    print(f"  Claims (wins): {len(claims)}")
    print(f"  Losses: {len(losses)}")
    print(f"  Sells: {len(sells)}")

    # Calculate P&L
    total_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in buys)
    total_won = sum(t['value'] if t['value'] > 0 else t['size'] for t in claims)
    total_sold = sum(t['value'] for t in sells)

    print(f"\n{'='*90}")
    print("FINANCIAL SUMMARY")
    print("=" * 90)
    print(f"  Total spent on buys: ${total_spent:.2f}")
    print(f"  Total from claims (wins): ${total_won:.2f}")
    print(f"  Total from sells: ${total_sold:.2f}")
    print(f"  NET P&L (72h): ${total_won + total_sold - total_spent:.2f}")

    # =========================================================================
    # ANALYSIS BY TIME PERIOD
    # =========================================================================
    print(f"\n{'='*90}")
    print("PERFORMANCE BY TIME PERIOD")
    print("=" * 90)

    for period_name, start_h, end_h in [("Last 24h", 0, 24), ("24-48h ago", 24, 48), ("48-72h ago", 48, 72)]:
        period_buys = [t for t in buys if start_h <= t['hours_ago'] < end_h]
        period_claims = [t for t in claims if start_h <= t['hours_ago'] < end_h]
        period_losses = [t for t in losses if start_h <= t['hours_ago'] < end_h]

        period_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in period_buys)
        period_won = sum(t['value'] if t['value'] > 0 else t['size'] for t in period_claims)

        print(f"\n{period_name}:")
        print(f"  Buys: {len(period_buys)} (${period_spent:.2f})")
        print(f"  Wins: {len(period_claims)} (${period_won:.2f})")
        print(f"  Losses: {len(period_losses)}")

    # =========================================================================
    # ANALYSIS BY OUTCOME TYPE (YES vs NO)
    # =========================================================================
    print(f"\n{'='*90}")
    print("PERFORMANCE BY OUTCOME TYPE")
    print("=" * 90)

    yes_buys = [t for t in buys if t['outcome'].lower() == 'yes']
    no_buys = [t for t in buys if t['outcome'].lower() == 'no']

    yes_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in yes_buys)
    no_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in no_buys)

    # Match claims to outcome type (harder - need to infer from title)
    yes_claims = [t for t in claims if 'no' not in t['title'].lower().split()[-5:]]  # Rough heuristic
    no_claims = [t for t in claims if t not in yes_claims]

    yes_won = sum(t['value'] if t['value'] > 0 else t['size'] for t in yes_claims)
    no_won = sum(t['value'] if t['value'] > 0 else t['size'] for t in no_claims)

    print(f"\nYES bets:")
    print(f"  Count: {len(yes_buys)}")
    print(f"  Total spent: ${yes_spent:.2f}")
    print(f"  Avg entry price: {(sum(t['price'] for t in yes_buys) / len(yes_buys) * 100):.1f}¢" if yes_buys else "  N/A")

    print(f"\nNO bets:")
    print(f"  Count: {len(no_buys)}")
    print(f"  Total spent: ${no_spent:.2f}")
    print(f"  Avg entry price: {(sum(t['price'] for t in no_buys) / len(no_buys) * 100):.1f}¢" if no_buys else "  N/A")

    # =========================================================================
    # ANALYSIS BY CITY
    # =========================================================================
    print(f"\n{'='*90}")
    print("PERFORMANCE BY CITY")
    print("=" * 90)

    city_stats = defaultdict(lambda: {'buys': 0, 'spent': 0, 'wins': 0, 'won': 0, 'losses': 0})

    for t in buys:
        city = t['city'] or 'Unknown'
        city_stats[city]['buys'] += 1
        city_stats[city]['spent'] += t['value'] if t['value'] > 0 else t['size'] * t['price']

    for t in claims:
        city = t['city'] or 'Unknown'
        city_stats[city]['wins'] += 1
        city_stats[city]['won'] += t['value'] if t['value'] > 0 else t['size']

    for t in losses:
        city = t['city'] or 'Unknown'
        city_stats[city]['losses'] += 1

    # Sort by net P&L
    city_list = []
    for city, stats in city_stats.items():
        net = stats['won'] - stats['spent']
        win_rate = stats['wins'] / (stats['wins'] + stats['losses']) * 100 if (stats['wins'] + stats['losses']) > 0 else 0
        city_list.append((city, stats['buys'], stats['spent'], stats['wins'], stats['won'], stats['losses'], net, win_rate))

    city_list.sort(key=lambda x: x[6], reverse=True)  # Sort by net P&L

    print(f"\n{'City':<12} {'Buys':>5} {'Spent':>8} {'Wins':>5} {'Won':>8} {'Losses':>6} {'Net':>8} {'Win%':>6}")
    print("-" * 70)
    for city, buys_n, spent, wins, won, losses_n, net, win_rate in city_list:
        status = "+" if net >= 0 else ""
        print(f"{city:<12} {buys_n:>5} ${spent:>7.2f} {wins:>5} ${won:>7.2f} {losses_n:>6} {status}${net:>7.2f} {win_rate:>5.1f}%")

    # =========================================================================
    # ANALYSIS BY ENTRY PRICE RANGE
    # =========================================================================
    print(f"\n{'='*90}")
    print("PERFORMANCE BY ENTRY PRICE")
    print("=" * 90)

    price_ranges = [
        ("< 5¢ (penny stocks)", 0, 0.05),
        ("5-10¢ (cheap)", 0.05, 0.10),
        ("10-20¢ (moderate)", 0.10, 0.20),
        ("20-40¢ (mid-range)", 0.20, 0.40),
        ("40-70¢ (expensive)", 0.40, 0.70),
        ("> 70¢ (very expensive)", 0.70, 1.01),
    ]

    for range_name, low, high in price_ranges:
        range_buys = [t for t in buys if low <= t['price'] < high]
        if not range_buys:
            continue

        range_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in range_buys)
        avg_price = sum(t['price'] for t in range_buys) / len(range_buys)

        print(f"\n{range_name}:")
        print(f"  Trades: {len(range_buys)}")
        print(f"  Total spent: ${range_spent:.2f}")
        print(f"  Avg price: {avg_price*100:.1f}¢")

    # =========================================================================
    # WINNING TRADES DETAILS
    # =========================================================================
    print(f"\n{'='*90}")
    print("WINNING TRADES (Claims)")
    print("=" * 90)

    for t in sorted(claims, key=lambda x: x['value'], reverse=True):
        value = t['value'] if t['value'] > 0 else t['size']
        print(f"\n  +${value:.2f} | {t['city'] or 'Unknown':<10} | {t['market_date'] or 'N/A'}")
        print(f"    {t['title'][:70]}")

    # =========================================================================
    # FILTER ANALYSIS
    # =========================================================================
    print(f"\n{'='*90}")
    print("FILTER IMPACT ANALYSIS")
    print("=" * 90)

    # NO bets
    no_bet_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in no_buys)
    print(f"\nNO bets (now disabled):")
    print(f"  Count: {len(no_buys)}")
    print(f"  Total spent: ${no_bet_spent:.2f}")
    print(f"  SAVINGS if disabled: ${no_bet_spent:.2f}")

    # Cheap YES bets (< 8%)
    cheap_yes = [t for t in yes_buys if t['price'] < 0.08]
    cheap_spent = sum(t['value'] if t['value'] > 0 else t['size'] * t['price'] for t in cheap_yes)
    print(f"\nCheap YES < 8¢ (now filtered):")
    print(f"  Count: {len(cheap_yes)}")
    print(f"  Total spent: ${cheap_spent:.2f}")
    print(f"  SAVINGS if filtered: ${cheap_spent:.2f}")

    # Combined
    print(f"\n{'='*90}")
    print(f"TOTAL POTENTIAL SAVINGS FROM NEW FILTERS: ${no_bet_spent + cheap_spent:.2f}")
    print("=" * 90)

    # =========================================================================
    # RAW TRADE LOG
    # =========================================================================
    print(f"\n{'='*90}")
    print("COMPLETE TRADE LOG (Last 72h)")
    print("=" * 90)

    for t in trades:
        hrs = t['hours_ago']
        if t['type'] in ['buy', 'bought']:
            cost = t['value'] if t['value'] > 0 else t['size'] * t['price']
            print(f"\n  [{hrs:5.1f}h ago] BUY {t['outcome']:3} @ {t['price']*100:.1f}¢ = ${cost:.2f}")
        elif t['type'] in ['claim', 'redeem']:
            value = t['value'] if t['value'] > 0 else t['size']
            print(f"\n  [{hrs:5.1f}h ago] WIN +${value:.2f}")
        elif t['type'] in ['lost', 'loss']:
            print(f"\n  [{hrs:5.1f}h ago] LOSS")
        elif t['type'] in ['sell', 'sold']:
            print(f"\n  [{hrs:5.1f}h ago] SELL ${t['value']:.2f}")

        print(f"    {t['city'] or 'Unknown':<10} | {t['title'][:60]}")


def extract_city(title: str) -> str | None:
    """Extract city name from market title."""
    title_lower = title.lower()
    cities = {
        'miami': 'Miami', 'atlanta': 'Atlanta', 'chicago': 'Chicago',
        'seattle': 'Seattle', 'london': 'London', 'nyc': 'NYC',
        'new york': 'NYC', 'toronto': 'Toronto', 'dallas': 'Dallas',
        'seoul': 'Seoul', 'ankara': 'Ankara', 'wellington': 'Wellington',
    }
    for key, name in cities.items():
        if key in title_lower:
            return name
    return None


def extract_bucket(title: str) -> str | None:
    """Extract temperature bucket from title."""
    # Match patterns like "68-69°F" or "5°C or higher"
    match = re.search(r'(\d+(?:-\d+)?)\s*°[FC]', title)
    if match:
        return match.group(0)
    return None


def extract_date(title: str) -> str | None:
    """Extract market date from title."""
    # Match patterns like "January 30" or "Jan 30"
    match = re.search(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})', title, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


if __name__ == "__main__":
    asyncio.run(analyze_72h())
