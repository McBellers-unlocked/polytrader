"""Trade logger - persists trade decisions with full forecast data for analysis."""

import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
from typing import Any

from src.logging import get_logger

logger = get_logger(__name__)

# Default log file location
TRADE_LOG_FILE = Path("trade_history.jsonl")


def log_trade(
    action: str,  # "EXECUTED", "SKIPPED", "FILTERED"
    city: str,
    market_date: str,
    outcome: str,
    side: str,
    # Price info
    market_price: float,
    fair_value: float,
    edge: float,
    # Forecast info
    forecast_mean: float,
    forecast_std: float,
    bucket_low: float | None,
    bucket_high: float | None,
    distance_from_mean: float,
    # Confidence info
    model_agreement: float,
    sample_size: int,
    # Trade details
    size_dollars: float = 0,
    order_id: str = "",
    # Filter info
    filter_reason: str = "",
    # Extra context
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a trade decision with full context for later analysis."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "city": city,
        "market_date": market_date,
        "outcome": outcome,
        "side": side,
        # Prices
        "market_price": round(market_price, 4),
        "fair_value": round(fair_value, 4),
        "edge": round(edge, 4),
        # Forecast
        "forecast_mean": round(forecast_mean, 2),
        "forecast_std": round(forecast_std, 2),
        "bucket_low": bucket_low,
        "bucket_high": bucket_high,
        "distance_from_mean": round(distance_from_mean, 2),
        # Confidence
        "model_agreement": round(model_agreement, 4),
        "sample_size": sample_size,
        # Trade
        "size_dollars": round(size_dollars, 2),
        "order_id": order_id,
        # Filter
        "filter_reason": filter_reason,
    }

    if extra:
        record["extra"] = extra

    # Append to JSONL file
    try:
        with open(TRADE_LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Trade logged", action=action, outcome=outcome, edge=edge)
    except Exception as e:
        logger.warning(f"Failed to log trade: {e}")


def log_trade_from_opportunity(
    action: str,
    opp: Any,  # TradingOpportunity
    filter_reason: str = "",
    order_id: str = "",
) -> None:
    """Log a trade from a TradingOpportunity object."""

    fv = opp.fair_value
    bucket_mid = None
    distance = 0.0

    if fv.low_bound is not None and fv.high_bound is not None:
        bucket_mid = (fv.low_bound + fv.high_bound) / 2
        distance = abs(bucket_mid - fv.kde_mean)

    log_trade(
        action=action,
        city=opp.market.city.name if opp.market.city else "unknown",
        market_date=opp.market.target_date.isoformat() if opp.market.target_date else "",
        outcome=opp.bucket.outcome,
        side=opp.side,
        market_price=float(opp.price),
        fair_value=fv.fair_probability,
        edge=opp.edge,
        forecast_mean=fv.kde_mean,
        forecast_std=fv.kde_std,
        bucket_low=fv.low_bound,
        bucket_high=fv.high_bound,
        distance_from_mean=distance,
        model_agreement=opp.model_agreement,
        sample_size=fv.sample_size,
        size_dollars=float(opp.suggested_size),
        order_id=order_id,
        filter_reason=filter_reason,
    )


def read_trade_log() -> list[dict]:
    """Read all trade records from the log file."""
    records = []
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def analyze_trade_log() -> None:
    """Analyze the trade log and print insights."""
    records = read_trade_log()

    if not records:
        print("No trade records found. Run the bot to generate trade history.")
        return

    executed = [r for r in records if r["action"] == "EXECUTED"]
    filtered = [r for r in records if r["action"] in ["FILTERED", "SKIPPED"]]

    print("=" * 90)
    print("TRADE LOG ANALYSIS (with forecast data)")
    print("=" * 90)
    print(f"\nTotal records: {len(records)}")
    print(f"  Executed: {len(executed)}")
    print(f"  Filtered/Skipped: {len(filtered)}")

    if executed:
        print(f"\n{'='*90}")
        print("EXECUTED TRADES")
        print("=" * 90)

        for t in executed:
            bucket = f"{t['bucket_low']}-{t['bucket_high']}" if t['bucket_low'] else t['outcome']
            print(f"\n  {t['timestamp'][:19]} | {t['city']:<10} | {bucket}")
            print(f"    Side: {t['side']} @ {t['market_price']*100:.1f}¢")
            print(f"    Fair Value: {t['fair_value']*100:.1f}% | Edge: {t['edge']*100:.1f}%")
            print(f"    Forecast: {t['forecast_mean']:.1f}° ± {t['forecast_std']:.1f}°")
            print(f"    Distance from mean: {t['distance_from_mean']:.1f}°")
            print(f"    Model agreement: {t['model_agreement']*100:.1f}%")
            print(f"    Size: ${t['size_dollars']:.2f}")

        # Aggregate stats
        avg_edge = sum(t['edge'] for t in executed) / len(executed)
        avg_distance = sum(t['distance_from_mean'] for t in executed) / len(executed)
        avg_agreement = sum(t['model_agreement'] for t in executed) / len(executed)

        print(f"\n{'='*90}")
        print("AGGREGATE STATS (Executed Trades)")
        print("=" * 90)
        print(f"  Avg Edge: {avg_edge*100:.1f}%")
        print(f"  Avg Distance from Mean: {avg_distance:.1f}°")
        print(f"  Avg Model Agreement: {avg_agreement*100:.1f}%")

    if filtered:
        print(f"\n{'='*90}")
        print("FILTERED TRADES (by reason)")
        print("=" * 90)

        from collections import Counter
        reasons = Counter(t.get('filter_reason', 'unknown') for t in filtered)
        for reason, count in reasons.most_common():
            print(f"  {count:3}x | {reason}")


if __name__ == "__main__":
    analyze_trade_log()
