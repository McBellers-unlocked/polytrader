"""Resolution tracker for updating prediction records with actual outcomes.

Periodically checks for markets that have closed and fetches
the actual temperature from Wunderground to resolve predictions.
"""

from datetime import date, datetime, timedelta
from typing import Any

from src.config import CITIES, CityConfig
from src.execution.datastore import DataStore
from src.logging import get_logger
from src.resolution.wunderground import WundergroundClient

logger = get_logger(__name__)


class ResolutionTracker:
    """
    Tracks market resolutions and updates prediction records.

    Runs periodically to:
    1. Find predictions for markets that should have resolved (target_date < today)
    2. Fetch actual temperature from Wunderground
    3. Determine winning outcome (which bucket the actual temp falls in)
    4. Update prediction records with actual outcome and P&L

    Usage:
        tracker = ResolutionTracker(datastore)
        resolved = await tracker.check_resolutions()
        print(f"Resolved {resolved} markets")
    """

    def __init__(
        self,
        datastore: DataStore,
        check_delay_hours: int = 6,
    ):
        """
        Initialize the resolution tracker.

        Args:
            datastore: DataStore instance for database access
            check_delay_hours: Hours after midnight to wait before checking
                              (gives Wunderground time to finalize data)
        """
        self.datastore = datastore
        self.check_delay_hours = check_delay_hours
        self._wunderground = WundergroundClient()
        self._last_check: datetime | None = None

    async def check_resolutions(
        self,
        force: bool = False,
    ) -> int:
        """
        Check for and resolve pending markets.

        Args:
            force: If True, check even if recently checked

        Returns:
            Number of markets resolved
        """
        # Don't check too frequently (every 6 hours by default)
        if not force and self._last_check:
            elapsed = (datetime.utcnow() - self._last_check).total_seconds()
            if elapsed < self.check_delay_hours * 3600:
                logger.debug(
                    "Skipping resolution check - too recent",
                    hours_since_last=elapsed / 3600,
                )
                return 0

        self._last_check = datetime.utcnow()

        # Only check markets that closed at least 6 hours ago
        # This gives Wunderground time to finalize the data
        check_before = date.today() - timedelta(days=1)

        # Get pending markets that need resolution
        pending = await self.datastore.get_markets_pending_resolution(
            before_date=check_before,
        )

        if not pending:
            logger.debug("No markets pending resolution")
            return 0

        logger.info(
            "Checking market resolutions",
            pending_count=len(pending),
        )

        resolved_count = 0
        await self._wunderground.connect()

        try:
            for market in pending:
                resolved = await self._resolve_market(market)
                if resolved:
                    resolved_count += 1

        finally:
            await self._wunderground.close()

        if resolved_count > 0:
            logger.info(
                "Resolved markets",
                count=resolved_count,
            )

        return resolved_count

    async def _resolve_market(self, market: dict[str, Any]) -> bool:
        """
        Resolve a single market by fetching actual temperature.

        Args:
            market: Dict with condition_id, city, target_date

        Returns:
            True if successfully resolved
        """
        city_name = market["city"]
        target_date = date.fromisoformat(market["target_date"])
        condition_id = market["condition_id"]

        # Find city config to get Wunderground URL
        city_config = self._find_city_config(city_name)
        if not city_config:
            logger.warning(
                "City config not found for resolution",
                city=city_name,
            )
            return False

        if not city_config.wunderground_url:
            logger.warning(
                "No Wunderground URL for city",
                city=city_name,
            )
            return False

        # Fetch actual temperature from Wunderground
        result = await self._wunderground.get_daily_high(
            wunderground_path=city_config.wunderground_url,
            target_date=target_date,
            unit=city_config.unit,
        )

        if not result:
            logger.warning(
                "Could not fetch Wunderground data for resolution",
                city=city_name,
                target_date=str(target_date),
            )
            return False

        actual_temp = result.high_temp

        # Determine winning outcome (which bucket the temp falls in)
        winning_outcome = await self._determine_winning_outcome(
            condition_id=condition_id,
            actual_temp=actual_temp,
        )

        if not winning_outcome:
            logger.warning(
                "Could not determine winning outcome",
                city=city_name,
                actual_temp=actual_temp,
            )
            return False

        # Save resolved market
        await self.datastore.save_resolved_market(
            condition_id=condition_id,
            city=city_name,
            target_date=target_date,
            actual_temperature=actual_temp,
            winning_outcome=winning_outcome,
            wunderground_url=result.source_url,
        )

        # Update all prediction records for this market
        await self._update_predictions(
            condition_id=condition_id,
            target_date=target_date,
            actual_temp=actual_temp,
            winning_outcome=winning_outcome,
        )

        logger.info(
            "Market resolved",
            city=city_name,
            target_date=str(target_date),
            actual_temp=actual_temp,
            winning_outcome=winning_outcome,
        )

        return True

    async def _determine_winning_outcome(
        self,
        condition_id: str,
        actual_temp: float,
    ) -> str | None:
        """
        Determine which bucket won based on actual temperature.

        Args:
            condition_id: Market condition ID
            actual_temp: Actual high temperature

        Returns:
            Winning outcome string (e.g., "55-60°F") or None
        """
        # Get all predictions for this market to find bucket bounds
        predictions = await self.datastore.get_resolved_predictions()

        # Filter to this condition_id and get unique buckets
        buckets = {}
        for pred in predictions:
            if pred.get("condition_id") == condition_id:
                outcome = pred.get("outcome", "")
                low = pred.get("bucket_low")
                high = pred.get("bucket_high")
                if outcome and outcome not in buckets:
                    buckets[outcome] = (low, high)

        # If no predictions found, try unresolved ones
        if not buckets:
            unresolved = await self.datastore.get_unresolved_predictions()
            for pred in unresolved:
                if pred.get("condition_id") == condition_id:
                    outcome = pred.get("outcome", "")
                    low = pred.get("bucket_low")
                    high = pred.get("bucket_high")
                    if outcome and outcome not in buckets:
                        buckets[outcome] = (low, high)

        if not buckets:
            return None

        # Find which bucket the actual temp falls into
        for outcome, (low, high) in buckets.items():
            # Handle edge cases for open-ended buckets
            if low is None:
                # Bucket like "<50°F"
                if high is not None and actual_temp < high:
                    return outcome
            elif high is None:
                # Bucket like "≥80°F"
                if actual_temp >= low:
                    return outcome
            else:
                # Normal bucket like "55-60°F"
                if low <= actual_temp < high:
                    return outcome

        # If no exact match, might be an edge case
        logger.warning(
            "Actual temp doesn't fit any bucket",
            actual_temp=actual_temp,
            buckets=list(buckets.keys()),
        )
        return None

    async def _update_predictions(
        self,
        condition_id: str,
        target_date: date,
        actual_temp: float,
        winning_outcome: str,
    ) -> int:
        """
        Update all prediction records for a resolved market.

        Args:
            condition_id: Market condition ID
            target_date: Target date of the market
            actual_temp: Actual high temperature
            winning_outcome: Which bucket won

        Returns:
            Number of predictions updated
        """
        # Get all unresolved predictions for this market
        all_predictions = await self.datastore.get_unresolved_predictions()

        updated = 0
        for pred in all_predictions:
            if pred.get("condition_id") != condition_id:
                continue
            if pred.get("target_date") != target_date.isoformat():
                continue

            pred_outcome = pred.get("outcome", "")
            prediction_correct = (pred_outcome == winning_outcome)

            # Calculate P&L if this was traded
            trade_pnl = None
            if pred.get("was_traded"):
                trade_price = pred.get("trade_price", 0)
                trade_size = pred.get("trade_size", 0)
                trade_side = pred.get("trade_side", "")

                if trade_side in ("BUY", "BUY_YES"):
                    # Bought YES: win if prediction correct
                    if prediction_correct:
                        trade_pnl = trade_size * (1 - trade_price)  # Profit
                    else:
                        trade_pnl = -trade_size * trade_price  # Loss
                elif trade_side in ("SELL", "BUY_NO"):
                    # Bought NO / Sold YES: win if prediction wrong
                    if not prediction_correct:
                        trade_pnl = trade_size * (1 - trade_price)  # Profit
                    else:
                        trade_pnl = -trade_size * trade_price  # Loss

            await self.datastore.resolve_prediction(
                prediction_id=pred["id"],
                actual_temperature=actual_temp,
                actual_outcome=winning_outcome,
                prediction_correct=prediction_correct,
                trade_pnl=trade_pnl,
            )
            updated += 1

        return updated

    def _find_city_config(self, city_name: str) -> CityConfig | None:
        """Find city config by name."""
        for key, config in CITIES.items():
            if config.name == city_name:
                return config
        return None


async def demo_resolution_tracker():
    """Demonstrate the resolution tracker."""
    from src.execution.datastore import DataStore

    print(f"\n{'='*60}")
    print("Resolution Tracker Demo")
    print(f"{'='*60}\n")

    async with DataStore() as datastore:
        tracker = ResolutionTracker(datastore)

        # Force check for resolutions
        resolved = await tracker.check_resolutions(force=True)
        print(f"Resolved {resolved} markets")

    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_resolution_tracker())
