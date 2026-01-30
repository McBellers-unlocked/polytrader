"""Backtest new trading filters on historical trades."""

import asyncio
import aiohttp
import re


async def backtest_filters():
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'

    async with aiohttp.ClientSession() as session:
        url = f'https://data-api.polymarket.com/positions?user={address}'
        async with session.get(url) as resp:
            positions = await resp.json()

    # Filter to weather positions only
    weather_keywords = ['temperature', 'highest', '°F', '°C']
    weather_positions = [
        pos for pos in positions
        if any(kw.lower() in (pos.get('title', '') or '').lower() for kw in weather_keywords)
    ]

    print("=" * 80)
    print("BACKTEST: NEW FILTERS ON HISTORICAL TRADES")
    print("=" * 80)
    print(f"\nTotal weather positions: {len(weather_positions)}")

    # New filter constants
    MIN_YES_PRICE = 0.08   # 8%
    MAX_YES_PRICE = 0.70   # 70%
    DISABLE_NO_BETS = True

    # Categorize trades
    original_trades = []
    filtered_trades = []  # Would pass new filters
    removed_trades = []   # Would be filtered out

    for pos in weather_positions:
        title = pos.get('title', '') or ''
        outcome = pos.get('outcome', 'Unknown')
        size = float(pos.get('size', 0) or 0)
        avg_price = float(pos.get('avgPrice', 0) or 0)
        cur_price = float(pos.get('curPrice', 0) or 0)
        realized_pnl = float(pos.get('realizedPnl', 0) or 0)

        cost = size * avg_price
        value = size * cur_price
        unrealized_pnl = value - cost
        total_pnl = unrealized_pnl + realized_pnl

        # Extract bucket info from title (e.g., "72-73°F" or "44-45°F")
        bucket_match = re.search(r'(\d+)-(\d+)°[FC]', title)
        bucket_low = int(bucket_match.group(1)) if bucket_match else None
        bucket_high = int(bucket_match.group(2)) if bucket_match else None
        bucket_mid = (bucket_low + bucket_high) / 2 if bucket_match else None

        trade_info = {
            'title': title,
            'outcome': outcome,
            'entry_price': avg_price,
            'cur_price': cur_price,
            'cost': cost,
            'pnl': total_pnl,
            'bucket_mid': bucket_mid,
        }
        original_trades.append(trade_info)

        # Apply filters
        filter_reason = None

        # Filter 1: Disable NO bets
        if DISABLE_NO_BETS and outcome == 'No':
            filter_reason = "NO bets disabled"

        # Filter 2: Min YES price
        elif outcome == 'Yes' and avg_price < MIN_YES_PRICE:
            filter_reason = f"YES price {avg_price*100:.1f}¢ < {MIN_YES_PRICE*100:.0f}¢ min"

        # Filter 3: Max YES price
        elif outcome == 'Yes' and avg_price > MAX_YES_PRICE:
            filter_reason = f"YES price {avg_price*100:.1f}¢ > {MAX_YES_PRICE*100:.0f}¢ max"

        if filter_reason:
            removed_trades.append({**trade_info, 'reason': filter_reason})
        else:
            filtered_trades.append(trade_info)

    # Calculate P&L for original vs filtered
    original_cost = sum(t['cost'] for t in original_trades)
    original_pnl = sum(t['pnl'] for t in original_trades)
    original_pct = (original_pnl / original_cost * 100) if original_cost > 0 else 0

    filtered_cost = sum(t['cost'] for t in filtered_trades)
    filtered_pnl = sum(t['pnl'] for t in filtered_trades)
    filtered_pct = (filtered_pnl / filtered_cost * 100) if filtered_cost > 0 else 0

    removed_cost = sum(t['cost'] for t in removed_trades)
    removed_pnl = sum(t['pnl'] for t in removed_trades)
    removed_pct = (removed_pnl / removed_cost * 100) if removed_cost > 0 else 0

    print(f"\n{'='*80}")
    print("ORIGINAL PERFORMANCE (ALL TRADES)")
    print("=" * 80)
    print(f"Trades: {len(original_trades)}")
    print(f"Total Cost: ${original_cost:.2f}")
    print(f"Total P&L: ${original_pnl:.2f} ({original_pct:+.1f}%)")

    print(f"\n{'='*80}")
    print("TRADES THAT WOULD BE FILTERED OUT")
    print("=" * 80)
    for t in removed_trades:
        status = "✓ WIN" if t['pnl'] > 0 else "✗ LOSS"
        print(f"\n{t['title'][:60]}...")
        print(f"  {t['outcome']} @ {t['entry_price']*100:.1f}¢ → P&L: ${t['pnl']:.2f} [{status}]")
        print(f"  Filter: {t['reason']}")

    print(f"\n  SUBTOTAL REMOVED:")
    print(f"  {len(removed_trades)} trades, ${removed_cost:.2f} cost, ${removed_pnl:.2f} P&L ({removed_pct:+.1f}%)")

    print(f"\n{'='*80}")
    print("SIMULATED PERFORMANCE WITH NEW FILTERS")
    print("=" * 80)
    print(f"Trades: {len(filtered_trades)} (removed {len(removed_trades)})")
    print(f"Total Cost: ${filtered_cost:.2f}")
    print(f"Total P&L: ${filtered_pnl:.2f} ({filtered_pct:+.1f}%)")

    print(f"\n{'='*80}")
    print("COMPARISON")
    print("=" * 80)
    print(f"  Original:  {len(original_trades)} trades, ${original_pnl:.2f} ({original_pct:+.1f}%)")
    print(f"  Filtered:  {len(filtered_trades)} trades, ${filtered_pnl:.2f} ({filtered_pct:+.1f}%)")
    print(f"  Improvement: ${filtered_pnl - original_pnl:.2f} ({filtered_pct - original_pct:+.1f} pp)")

    # Breakdown of kept trades
    if filtered_trades:
        print(f"\n{'='*80}")
        print("TRADES THAT WOULD BE KEPT")
        print("=" * 80)
        for t in sorted(filtered_trades, key=lambda x: x['pnl'], reverse=True):
            status = "✓ WIN" if t['pnl'] > 0 else "✗ LOSS"
            print(f"\n{t['title'][:60]}...")
            print(f"  {t['outcome']} @ {t['entry_price']*100:.1f}¢ → P&L: ${t['pnl']:.2f} [{status}]")

    # Note about distance filter
    print(f"\n{'='*80}")
    print("NOTE: DISTANCE FILTER NOT SIMULATED")
    print("=" * 80)
    print("The 'within 3°F of forecast mean' filter cannot be backtested here")
    print("because we don't have the historical forecast data for each trade.")
    print("However, cheap YES bets (<8¢) are typically on buckets far from the mean,")
    print("so the MIN_YES_PRICE filter captures most of those bad trades.")


if __name__ == "__main__":
    asyncio.run(backtest_filters())
