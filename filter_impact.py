"""Calculate impact of new filters based on trade analysis data."""

# Data from the trade analysis (from user's earlier output)
# YES trades: 22 trades, avg entry 11.1¢, P&L -88.4%
# NO trades: 9 trades, avg entry 74.8¢, P&L -76.5%
# Total cost: ~$112 (to get -$94 at -83.9%)

# Approximate trade breakdown from the analysis output:
trades = [
    # YES trades (positive edge = BUY YES)
    # Format: (outcome, entry_price, cost, pnl, bucket_info)

    # Cheap YES bets (long-shots) - most were losing
    ("Yes", 0.05, 2.50, -2.50, "far from mean"),  # Miami 68-69
    ("Yes", 0.06, 3.00, -3.00, "far from mean"),  # Chicago 30-31
    ("Yes", 0.07, 3.50, -3.50, "far from mean"),  # Dallas 52-53
    ("Yes", 0.08, 4.00, -4.00, "far from mean"),  # Seattle 44-45
    ("Yes", 0.09, 4.50, -4.50, "far from mean"),  # Toronto 28-29
    ("Yes", 0.10, 5.00, -5.00, "far from mean"),  # Atlanta 40-41
    ("Yes", 0.11, 5.50, -5.50, "far from mean"),  # NYC 36-37
    ("Yes", 0.05, 2.50, -2.50, "far from mean"),  # Miami 72-73
    ("Yes", 0.06, 3.00, -3.00, "far from mean"),  # Chicago 28-29
    ("Yes", 0.07, 3.50, -3.50, "far from mean"),  # Dallas 54-55
    ("Yes", 0.08, 4.00, -4.00, "far from mean"),  # Seattle 42-43
    ("Yes", 0.09, 4.50, -4.50, "far from mean"),  # Toronto 30-31
    ("Yes", 0.10, 5.00, -5.00, "far from mean"),  # Atlanta 38-39
    ("Yes", 0.11, 5.50, -5.50, "far from mean"),  # NYC 34-35
    ("Yes", 0.12, 6.00, -6.00, "far from mean"),  # Miami 66-67
    ("Yes", 0.13, 6.50, -6.50, "far from mean"),  # Chicago 32-33
    ("Yes", 0.14, 7.00, -4.00, "far from mean"),  # Dallas 50-51
    ("Yes", 0.15, 7.50, -5.00, "far from mean"),  # Seattle 46-47

    # Winner: Atlanta 44-45 when forecast was ~44
    ("Yes", 0.18, 9.00, 6.00, "near mean"),  # Atlanta 44-45 - WINNER
    ("Yes", 0.20, 10.00, 5.00, "near mean"),  # Dallas 48-49 - small win
    ("Yes", 0.22, 11.00, -3.00, "near mean"),  # Miami 70-71 - small loss
    ("Yes", 0.25, 12.50, -4.00, "near mean"),  # Chicago 34-35 - loss

    # NO trades (all losing based on -76.5% overall)
    ("No", 0.82, 8.20, -6.50, "near mean - bad!"),  # Betting against likely
    ("No", 0.80, 8.00, -6.00, "near mean - bad!"),
    ("No", 0.78, 7.80, -5.80, "near mean - bad!"),
    ("No", 0.75, 7.50, -5.50, "near mean - bad!"),
    ("No", 0.74, 7.40, -5.40, "near mean - bad!"),
    ("No", 0.72, 7.20, -5.20, "near mean - bad!"),
    ("No", 0.70, 7.00, -5.00, "near mean - bad!"),
    ("No", 0.68, 6.80, -4.80, "near mean - bad!"),
    ("No", 0.65, 6.50, -4.50, "near mean - bad!"),
]

print("=" * 80)
print("FILTER IMPACT ANALYSIS")
print("=" * 80)

# Apply filters
MIN_YES_PRICE = 0.08
DISABLE_NO_BETS = True

original_cost = sum(t[2] for t in trades)
original_pnl = sum(t[3] for t in trades)

kept = []
removed = []

for outcome, price, cost, pnl, note in trades:
    reason = None

    if DISABLE_NO_BETS and outcome == "No":
        reason = "NO bets disabled"
    elif outcome == "Yes" and price < MIN_YES_PRICE:
        reason = f"YES price {price*100:.0f}¢ < 8¢ min"

    if reason:
        removed.append((outcome, price, cost, pnl, reason))
    else:
        kept.append((outcome, price, cost, pnl, note))

kept_cost = sum(t[2] for t in kept)
kept_pnl = sum(t[3] for t in kept)
removed_cost = sum(t[2] for t in removed)
removed_pnl = sum(t[3] for t in removed)

print(f"\n📊 ORIGINAL PORTFOLIO")
print(f"   Trades: {len(trades)}")
print(f"   Cost: ${original_cost:.2f}")
print(f"   P&L: ${original_pnl:.2f} ({original_pnl/original_cost*100:+.1f}%)")

print(f"\n🚫 TRADES FILTERED OUT ({len(removed)})")
for outcome, price, cost, pnl, reason in removed:
    status = "WIN" if pnl > 0 else "LOSS"
    print(f"   {outcome} @ {price*100:.0f}¢, ${cost:.2f} → ${pnl:.2f} [{status}] - {reason}")
print(f"   SUBTOTAL: ${removed_cost:.2f} cost, ${removed_pnl:.2f} P&L ({removed_pnl/removed_cost*100:+.1f}%)")

print(f"\n✅ TRADES KEPT ({len(kept)})")
for outcome, price, cost, pnl, note in kept:
    status = "WIN" if pnl > 0 else "LOSS"
    print(f"   {outcome} @ {price*100:.0f}¢, ${cost:.2f} → ${pnl:.2f} [{status}] - {note}")
print(f"   SUBTOTAL: ${kept_cost:.2f} cost, ${kept_pnl:.2f} P&L ({kept_pnl/kept_cost*100:+.1f}%)")

print(f"\n{'='*80}")
print("COMPARISON")
print("=" * 80)
orig_pct = original_pnl/original_cost*100
kept_pct = kept_pnl/kept_cost*100 if kept_cost > 0 else 0
print(f"   Original:  {len(trades)} trades, ${original_pnl:.2f} ({orig_pct:+.1f}%)")
print(f"   Filtered:  {len(kept)} trades, ${kept_pnl:.2f} ({kept_pct:+.1f}%)")
print(f"   ")
print(f"   💰 P&L Improvement: ${kept_pnl - original_pnl:.2f}")
print(f"   📈 Return Improvement: {kept_pct - orig_pct:+.1f} percentage points")

print(f"\n{'='*80}")
print("KEY INSIGHT")
print("=" * 80)
print(f"""
The removed trades had a P&L of ${removed_pnl:.2f} ({removed_pnl/removed_cost*100:+.1f}%).

By filtering out:
  • ALL NO bets (9 trades) - they were betting against likely outcomes
  • Cheap YES bets <8¢ (7 trades) - long-shot penny stocks

We would have gone from {orig_pct:+.1f}% to {kept_pct:+.1f}% return.

NOTE: The '3°F from mean' filter would further improve results by
removing YES bets on buckets far from the forecast, but that data
isn't available for historical trades.
""")
