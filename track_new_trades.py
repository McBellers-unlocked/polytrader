"""Track P&L for trades made after a specific date (when new filters were applied)."""

import asyncio
import aiohttp
from datetime import datetime, timezone

# Set this to when you deployed the new filters
FILTER_DEPLOY_DATE = "2025-01-30T00:00:00Z"  # Adjust this date


async def track_trades():
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'
    cutoff = datetime.fromisoformat(FILTER_DEPLOY_DATE.replace('Z', '+00:00'))

    async with aiohttp.ClientSession() as session:
        # Fetch activity (trades)
        url = f'https://data-api.polymarket.com/activity?user={address}&limit=200'
        async with session.get(url) as resp:
            activity = await resp.json()

    # Filter to weather trades only
    weather_keywords = ['temperature', 'highest', '°F', '°C']

    old_trades = []
    new_trades = []

    for trade in activity:
        title = trade.get('title', '') or trade.get('market', '') or ''
        if not any(kw.lower() in title.lower() for kw in weather_keywords):
            continue

        # Parse timestamp
        timestamp_str = trade.get('timestamp') or trade.get('createdAt') or ''
        if timestamp_str:
            try:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                is_new = ts >= cutoff
            except:
                is_new = False
        else:
            is_new = False

        trade_info = {
            'title': title[:60],
            'type': trade.get('type', ''),
            'side': trade.get('side', ''),
            'outcome': trade.get('outcome', ''),
            'price': float(trade.get('price', 0) or 0),
            'size': float(trade.get('size', 0) or 0),
            'value': float(trade.get('value', 0) or 0),
            'timestamp': timestamp_str,
        }

        if is_new:
            new_trades.append(trade_info)
        else:
            old_trades.append(trade_info)

    print("=" * 80)
    print(f"TRADE TRACKING - Cutoff: {FILTER_DEPLOY_DATE}")
    print("=" * 80)

    print(f"\n📊 OLD TRADES (before filters): {len(old_trades)}")
    print(f"📊 NEW TRADES (after filters): {len(new_trades)}")

    if new_trades:
        print(f"\n{'='*80}")
        print("NEW TRADES (after filter deployment)")
        print("=" * 80)
        for t in new_trades:
            print(f"  {t['type']:8} | {t['outcome']:3} @ {t['price']*100:.0f}¢ | {t['title']}")


async def analyze_from_positions():
    """Alternative: analyze based on entry dates inferred from market titles."""
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'

    async with aiohttp.ClientSession() as session:
        url = f'https://data-api.polymarket.com/positions?user={address}'
        async with session.get(url) as resp:
            positions = await resp.json()

    weather_keywords = ['temperature', 'highest', '°F', '°C']

    # Group by market date
    jan_29 = []
    jan_30 = []
    jan_31 = []
    older = []

    for pos in positions:
        title = pos.get('title', '') or ''
        if not any(kw.lower() in title.lower() for kw in weather_keywords):
            continue

        outcome = pos.get('outcome', '')
        size = float(pos.get('size', 0) or 0)
        avg_price = float(pos.get('avgPrice', 0) or 0)
        cur_price = float(pos.get('curPrice', 0) or 0)
        realized = float(pos.get('realizedPnl', 0) or 0)

        cost = size * avg_price
        value = size * cur_price
        pnl = value - cost + realized

        trade = {
            'title': title[:55],
            'outcome': outcome,
            'entry': avg_price,
            'cost': cost,
            'pnl': pnl,
        }

        if 'January 31' in title or 'Jan 31' in title:
            jan_31.append(trade)
        elif 'January 30' in title or 'Jan 30' in title:
            jan_30.append(trade)
        elif 'January 29' in title or 'Jan 29' in title:
            jan_29.append(trade)
        else:
            older.append(trade)

    print("\n" + "=" * 80)
    print("P&L BY MARKET DATE")
    print("=" * 80)

    for label, trades in [("Older", older), ("Jan 29", jan_29), ("Jan 30", jan_30), ("Jan 31", jan_31)]:
        if trades:
            total_cost = sum(t['cost'] for t in trades)
            total_pnl = sum(t['pnl'] for t in trades)
            pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            print(f"\n{label} Markets: {len(trades)} positions")
            print(f"  Cost: ${total_cost:.2f}")
            print(f"  P&L: ${total_pnl:.2f} ({pct:+.1f}%)")

            # Show individual trades
            for t in sorted(trades, key=lambda x: x['pnl'], reverse=True):
                status = "WIN" if t['pnl'] > 0 else "LOSS" if t['pnl'] < 0 else "OPEN"
                print(f"    {t['outcome']:3} @ {t['entry']*100:5.1f}¢ | ${t['pnl']:+6.2f} [{status}] | {t['title']}")

    # Summary
    all_trades = older + jan_29 + jan_30 + jan_31
    total_cost = sum(t['cost'] for t in all_trades)
    total_pnl = sum(t['pnl'] for t in all_trades)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total positions: {len(all_trades)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Total P&L: ${total_pnl:.2f} ({total_pnl/total_cost*100:+.1f}%)")

    # Highlight what new filters would have saved
    no_trades = [t for t in all_trades if t['outcome'] == 'No']
    cheap_yes = [t for t in all_trades if t['outcome'] == 'Yes' and t['entry'] < 0.08]

    no_pnl = sum(t['pnl'] for t in no_trades)
    cheap_pnl = sum(t['pnl'] for t in cheap_yes)

    print(f"\n{'='*80}")
    print("FILTER IMPACT (what new rules would have saved)")
    print("=" * 80)
    print(f"NO bets (now disabled): {len(no_trades)} trades, ${no_pnl:.2f}")
    print(f"Cheap YES <8¢ (now filtered): {len(cheap_yes)} trades, ${cheap_pnl:.2f}")
    print(f"Combined savings: ${-(no_pnl + cheap_pnl):.2f}")


if __name__ == "__main__":
    asyncio.run(analyze_from_positions())
