"""Analyze weather trading performance over the last 48 hours."""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import defaultdict
import json


async def fetch_and_analyze():
    # Your funder address
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'

    async with aiohttp.ClientSession() as session:
        # Fetch positions
        url = f'https://data-api.polymarket.com/positions?user={address}'
        async with session.get(url) as resp:
            positions = await resp.json()

        # Fetch activity/trades
        activity_url = f'https://data-api.polymarket.com/activity?user={address}&limit=100'
        async with session.get(activity_url) as resp:
            activity = await resp.json()

        print("=" * 80)
        print("POLYTRADER TRADE ANALYSIS - LAST 48 HOURS")
        print("=" * 80)
        print(f"\nTotal positions: {len(positions)}")
        print(f"Recent activities: {len(activity)}")

        # Categorize positions
        weather_positions = []
        other_positions = []

        weather_keywords = ['temperature', 'highest', '°F', '°C', 'weather']

        for pos in positions:
            title = pos.get('title', '') or pos.get('market', {}).get('question', '') or ''
            is_weather = any(kw.lower() in title.lower() for kw in weather_keywords)

            if is_weather:
                weather_positions.append(pos)
            else:
                other_positions.append(pos)

        # Analyze weather positions
        print(f"\n{'='*80}")
        print(f"WEATHER POSITIONS ({len(weather_positions)})")
        print("=" * 80)

        total_cost = 0
        total_value = 0
        total_realized = 0

        by_outcome_type = defaultdict(list)  # YES vs NO
        by_city = defaultdict(list)
        by_price_range = defaultdict(list)

        for pos in weather_positions:
            title = pos.get('title', '') or pos.get('market', {}).get('question', '') or 'Unknown'
            outcome = pos.get('outcome', 'Unknown')
            size = float(pos.get('size', 0) or 0)
            avg_price = float(pos.get('avgPrice', 0) or 0)
            cur_price = float(pos.get('curPrice', 0) or 0)
            realized_pnl = float(pos.get('realizedPnl', 0) or 0)

            cost = size * avg_price
            value = size * cur_price
            unrealized_pnl = value - cost
            pnl_pct = (unrealized_pnl / cost * 100) if cost > 0 else 0

            total_cost += cost
            total_value += value
            total_realized += realized_pnl

            # Categorize
            by_outcome_type[outcome].append({
                'title': title,
                'cost': cost,
                'value': value,
                'pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'entry_price': avg_price,
                'cur_price': cur_price,
                'realized': realized_pnl,
            })

            # Extract city
            for city in ['Miami', 'Atlanta', 'Chicago', 'Dallas', 'Seattle', 'Toronto', 'London', 'Seoul', 'NYC', 'New York']:
                if city.lower() in title.lower():
                    by_city[city].append({
                        'title': title,
                        'pnl': unrealized_pnl + realized_pnl,
                        'cost': cost,
                    })
                    break

            # Categorize by entry price
            if avg_price < 0.20:
                by_price_range['<20%'].append({'pnl': unrealized_pnl, 'cost': cost})
            elif avg_price < 0.50:
                by_price_range['20-50%'].append({'pnl': unrealized_pnl, 'cost': cost})
            elif avg_price < 0.70:
                by_price_range['50-70%'].append({'pnl': unrealized_pnl, 'cost': cost})
            else:
                by_price_range['>70%'].append({'pnl': unrealized_pnl, 'cost': cost})

            # Print position details
            status = "+" if unrealized_pnl >= 0 else ""
            print(f"\n{title[:70]}...")
            print(f"  {outcome} | Entry: {avg_price*100:.1f}¢ → Current: {cur_price*100:.1f}¢")
            print(f"  Cost: ${cost:.2f} | Value: ${value:.2f} | P&L: {status}${unrealized_pnl:.2f} ({pnl_pct:+.1f}%)")
            if realized_pnl != 0:
                print(f"  Realized P&L: ${realized_pnl:.2f}")

        # Summary statistics
        total_pnl = total_value - total_cost + total_realized
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal Weather Positions: {len(weather_positions)}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Current Value: ${total_value:.2f}")
        print(f"Unrealized P&L: ${total_value - total_cost:.2f}")
        print(f"Realized P&L: ${total_realized:.2f}")
        print(f"TOTAL P&L: ${total_pnl:.2f} ({total_pnl_pct:+.1f}%)")

        # Analysis by outcome type (YES vs NO)
        print(f"\n{'='*80}")
        print("PERFORMANCE BY OUTCOME TYPE")
        print("=" * 80)

        for outcome_type, trades in by_outcome_type.items():
            total_type_cost = sum(t['cost'] for t in trades)
            total_type_pnl = sum(t['pnl'] + t['realized'] for t in trades)
            type_pnl_pct = (total_type_pnl / total_type_cost * 100) if total_type_cost > 0 else 0

            print(f"\n{outcome_type} positions: {len(trades)}")
            print(f"  Total Cost: ${total_type_cost:.2f}")
            print(f"  Total P&L: ${total_type_pnl:.2f} ({type_pnl_pct:+.1f}%)")

            # Show avg entry price
            avg_entry = sum(t['entry_price'] for t in trades) / len(trades) if trades else 0
            print(f"  Avg Entry Price: {avg_entry*100:.1f}¢")

        # Analysis by city
        print(f"\n{'='*80}")
        print("PERFORMANCE BY CITY")
        print("=" * 80)

        city_performance = []
        for city, trades in by_city.items():
            total_city_cost = sum(t['cost'] for t in trades)
            total_city_pnl = sum(t['pnl'] for t in trades)
            city_pnl_pct = (total_city_pnl / total_city_cost * 100) if total_city_cost > 0 else 0
            city_performance.append((city, len(trades), total_city_cost, total_city_pnl, city_pnl_pct))

        # Sort by P&L %
        city_performance.sort(key=lambda x: x[4], reverse=True)

        for city, count, cost, pnl, pnl_pct in city_performance:
            status = "+" if pnl >= 0 else ""
            print(f"  {city}: {count} trades, ${cost:.2f} cost, {status}${pnl:.2f} ({pnl_pct:+.1f}%)")

        # Analysis by entry price range
        print(f"\n{'='*80}")
        print("PERFORMANCE BY ENTRY PRICE")
        print("=" * 80)

        for price_range in ['<20%', '20-50%', '50-70%', '>70%']:
            trades = by_price_range.get(price_range, [])
            if trades:
                total_cost_range = sum(t['cost'] for t in trades)
                total_pnl_range = sum(t['pnl'] for t in trades)
                pnl_pct_range = (total_pnl_range / total_cost_range * 100) if total_cost_range > 0 else 0
                print(f"  Entry {price_range}: {len(trades)} trades, ${total_cost_range:.2f} cost, ${total_pnl_range:.2f} ({pnl_pct_range:+.1f}%)")

        # Key insights
        print(f"\n{'='*80}")
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)

        # Check if NO bets at high prices are losing
        no_trades = by_outcome_type.get('No', [])
        high_price_no = [t for t in no_trades if t['entry_price'] > 0.70]
        if high_price_no:
            high_price_no_pnl = sum(t['pnl'] for t in high_price_no)
            high_price_no_cost = sum(t['cost'] for t in high_price_no)
            print(f"\n1. HIGH-PRICE NO BETS (>70¢):")
            print(f"   {len(high_price_no)} trades, P&L: ${high_price_no_pnl:.2f} ({high_price_no_pnl/high_price_no_cost*100:.1f}%)")
            if high_price_no_pnl < 0:
                print("   RECOMMENDATION: Avoid buying NO above 70¢ - bad risk/reward")

        # Check YES vs NO performance
        yes_trades = by_outcome_type.get('Yes', [])
        if yes_trades and no_trades:
            yes_pnl = sum(t['pnl'] for t in yes_trades)
            no_pnl = sum(t['pnl'] for t in no_trades)
            print(f"\n2. YES vs NO PERFORMANCE:")
            print(f"   YES trades: {len(yes_trades)}, P&L: ${yes_pnl:.2f}")
            print(f"   NO trades: {len(no_trades)}, P&L: ${no_pnl:.2f}")
            if no_pnl < yes_pnl:
                print("   RECOMMENDATION: YES bets performing better than NO bets")

        # Worst performing positions
        all_trades = []
        for trades in by_outcome_type.values():
            all_trades.extend(trades)

        worst = sorted(all_trades, key=lambda x: x['pnl_pct'])[:3]
        print(f"\n3. WORST PERFORMERS:")
        for t in worst:
            print(f"   {t['title'][:50]}...")
            print(f"   Entry: {t['entry_price']*100:.1f}¢, P&L: ${t['pnl']:.2f} ({t['pnl_pct']:+.1f}%)")

        # Best performing
        best = sorted(all_trades, key=lambda x: x['pnl_pct'], reverse=True)[:3]
        print(f"\n4. BEST PERFORMERS:")
        for t in best:
            print(f"   {t['title'][:50]}...")
            print(f"   Entry: {t['entry_price']*100:.1f}¢, P&L: ${t['pnl']:.2f} ({t['pnl_pct']:+.1f}%)")

        print(f"\n{'='*80}")
        print("Run this script periodically to track performance.")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(fetch_and_analyze())
