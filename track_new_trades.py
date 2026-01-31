"""Track P&L for trades including realized wins from activity history."""

import asyncio
import aiohttp
from datetime import datetime, timezone
from collections import defaultdict


async def fetch_full_pnl():
    """Fetch both positions AND activity to get complete P&L picture."""
    address = '0x12b065A1232b26Af79dB55b9e71F5cffEBCaCD23'

    async with aiohttp.ClientSession() as session:
        # Fetch current positions (open trades)
        pos_url = f'https://data-api.polymarket.com/positions?user={address}'
        async with session.get(pos_url) as resp:
            positions = await resp.json()

        # Fetch activity history (includes claims, buys, sells)
        act_url = f'https://data-api.polymarket.com/activity?user={address}&limit=500'
        async with session.get(act_url) as resp:
            activity = await resp.json()

    weather_keywords = ['temperature', 'highest', '°F', '°C']

    print("=" * 80)
    print("COMPLETE P&L ANALYSIS (Positions + Realized)")
    print("=" * 80)

    # Track realized P&L from activity
    realized_wins = []
    realized_losses = []
    buys = []

    for act in activity:
        title = act.get('title', '') or act.get('description', '') or ''
        if not any(kw.lower() in title.lower() for kw in weather_keywords):
            continue

        act_type = act.get('type', '').lower()
        value = float(act.get('value', 0) or act.get('usdcSize', 0) or 0)
        price = float(act.get('price', 0) or 0)
        size = float(act.get('size', 0) or 0)

        if act_type in ['claim', 'redeem', 'claimed']:
            # This is a WIN - market resolved in our favor
            realized_wins.append({
                'title': title[:55],
                'value': value if value > 0 else size,  # payout amount
                'type': act_type,
            })
        elif act_type in ['sell', 'sold']:
            # Sold position - could be profit or loss
            realized_wins.append({
                'title': title[:55],
                'value': value,
                'type': 'sell',
            })
        elif act_type in ['buy', 'bought']:
            buys.append({
                'title': title[:55],
                'cost': value if value > 0 else (size * price),
                'price': price,
            })
        elif act_type in ['lost', 'loss']:
            realized_losses.append({
                'title': title[:55],
                'type': 'lost',
            })

    # Calculate realized P&L
    total_wins = sum(w['value'] for w in realized_wins)
    total_bought = sum(b['cost'] for b in buys)

    print(f"\n📈 REALIZED WINS (Claims + Sells):")
    print("-" * 60)
    for w in realized_wins:
        print(f"  +${w['value']:.2f} | {w['type']:6} | {w['title']}")
    print(f"  TOTAL REALIZED WINS: ${total_wins:.2f}")

    print(f"\n📉 TOTAL SPENT ON BUYS: ${total_bought:.2f}")

    # Current open positions
    print(f"\n📊 CURRENT OPEN POSITIONS:")
    print("-" * 60)

    open_cost = 0
    open_value = 0
    for pos in positions:
        title = pos.get('title', '') or ''
        if not any(kw.lower() in title.lower() for kw in weather_keywords):
            continue

        outcome = pos.get('outcome', '')
        size = float(pos.get('size', 0) or 0)
        avg_price = float(pos.get('avgPrice', 0) or 0)
        cur_price = float(pos.get('curPrice', 0) or 0)

        cost = size * avg_price
        value = size * cur_price
        pnl = value - cost

        open_cost += cost
        open_value += value

        status = "+" if pnl >= 0 else ""
        print(f"  {outcome:3} @ {avg_price*100:5.1f}¢ → {cur_price*100:5.1f}¢ | {status}${pnl:.2f} | {title[:45]}")

    print(f"\n  Open positions cost: ${open_cost:.2f}")
    print(f"  Open positions value: ${open_value:.2f}")
    print(f"  Open positions P&L: ${open_value - open_cost:.2f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total spent (buys): ${total_bought:.2f}")
    print(f"  Total realized (wins/sells): ${total_wins:.2f}")
    print(f"  Open position value: ${open_value:.2f}")
    print(f"  ")
    net_pnl = total_wins + open_value - total_bought
    print(f"  NET P&L: ${net_pnl:.2f}")


async def track_trades():
    """Legacy function - redirects to full P&L analysis."""
    await fetch_full_pnl()


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
    asyncio.run(fetch_full_pnl())
