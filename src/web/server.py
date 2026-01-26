"""FastAPI web server for Polytrader dashboard.

Provides a web interface to monitor the trading bot.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
import math

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import get_settings, CITIES, TradingMode
from src.execution.datastore import DataStore
from src.risk.risk_manager import RiskManager
from src.main import TradingBot
from src.markets.client import PolymarketClient

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Global bot instance (optional - for running bot from web)
_bot: TradingBot | None = None
_bot_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown
    global _bot, _bot_task
    if _bot:
        await _bot.stop()
    if _bot_task:
        _bot_task.cancel()


app = FastAPI(
    title="Polytrader Dashboard",
    description="Web dashboard for Polymarket weather trading bot",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =============================================================================
# Helper Functions
# =============================================================================

async def get_dashboard_data() -> dict[str, Any]:
    """Get all dashboard data."""
    settings = get_settings()

    # Risk status
    risk = RiskManager(starting_bankroll=Decimal(str(settings.starting_bankroll)))
    risk_status = risk.get_status()

    # Database data
    datastore = DataStore()
    await datastore.connect()

    try:
        today = date.today()

        # Open positions
        positions = await datastore.get_open_positions()

        # Today's trades
        trades = await datastore.get_trades(start_date=today, end_date=today)

        # Recent opportunities
        opportunities = await datastore.get_opportunities(
            start_date=today,
            end_date=today,
            min_edge=0.05,
        )
        opportunities.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

        # Weekly summaries
        week_ago = today - timedelta(days=7)
        summaries = await datastore.get_daily_summaries(
            start_date=week_ago,
            end_date=today,
        )

        # Stats
        stats = await datastore.get_statistics()

    finally:
        await datastore.close()

    return {
        "settings": {
            "mode": settings.trading_mode.value,
            "bankroll": settings.starting_bankroll,
            "min_edge": settings.min_edge_threshold,
            "min_agreement": settings.min_model_agreement,
        },
        "risk": risk_status,
        "positions": positions[:10],
        "trades": trades[:20],
        "opportunities": opportunities[:15],
        "summaries": summaries,
        "stats": stats,
        "bot_running": _bot is not None and _bot._running,
        "timestamp": datetime.now().isoformat(),
    }


async def get_ml_analytics_data() -> dict[str, Any]:
    """Get ML training analytics data."""
    datastore = DataStore()
    await datastore.connect()

    try:
        today = date.today()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)

        # Get resolved predictions for calibration analysis
        resolved = await datastore.get_resolved_predictions(
            start_date=month_ago,
            end_date=today,
        )

        # Get unresolved predictions (pending)
        unresolved = await datastore.get_unresolved_predictions(before_date=today)

        # Calculate calibration buckets (10% increments)
        calibration = []
        for i in range(10):
            low = i * 0.1
            high = (i + 1) * 0.1
            bucket_preds = [
                p for p in resolved
                if low <= p.get("model_probability", 0) < high
            ]
            if bucket_preds:
                n_correct = sum(1 for p in bucket_preds if p.get("prediction_correct"))
                actual_rate = n_correct / len(bucket_preds)
                calibration.append({
                    "bucket": f"{int(low*100)}-{int(high*100)}%",
                    "midpoint": (low + high) / 2,
                    "count": len(bucket_preds),
                    "correct": n_correct,
                    "actual_rate": actual_rate,
                    "expected_rate": (low + high) / 2,
                    "calibration_error": abs(actual_rate - (low + high) / 2),
                })

        # Edge vs win rate analysis
        edge_buckets = []
        for edge_low in [0.10, 0.15, 0.20, 0.30, 0.50]:
            edge_high = edge_low + 0.10 if edge_low < 0.50 else 1.0
            bucket_preds = [
                p for p in resolved
                if p.get("was_traded") and edge_low <= abs(p.get("calculated_edge", 0)) < edge_high
            ]
            if bucket_preds:
                n_correct = sum(1 for p in bucket_preds if p.get("prediction_correct"))
                total_pnl = sum(p.get("trade_pnl", 0) or 0 for p in bucket_preds)
                edge_buckets.append({
                    "bucket": f"{int(edge_low*100)}%+",
                    "count": len(bucket_preds),
                    "win_rate": n_correct / len(bucket_preds) * 100,
                    "total_pnl": total_pnl,
                    "avg_pnl": total_pnl / len(bucket_preds) if bucket_preds else 0,
                })

        # City performance
        city_stats = {}
        for p in resolved:
            city = p.get("city", "Unknown")
            if city not in city_stats:
                city_stats[city] = {"total": 0, "correct": 0, "traded": 0, "pnl": 0}
            city_stats[city]["total"] += 1
            if p.get("prediction_correct"):
                city_stats[city]["correct"] += 1
            if p.get("was_traded"):
                city_stats[city]["traded"] += 1
                city_stats[city]["pnl"] += p.get("trade_pnl", 0) or 0

        city_performance = [
            {
                "city": city,
                "predictions": stats["total"],
                "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                "trades": stats["traded"],
                "pnl": stats["pnl"],
            }
            for city, stats in city_stats.items()
        ]
        city_performance.sort(key=lambda x: x["predictions"], reverse=True)

        # Recent predictions (last 50)
        recent_predictions = sorted(
            resolved + unresolved,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:50]

        # Summary stats
        total_predictions = len(resolved) + len(unresolved)
        resolved_count = len(resolved)
        if resolved:
            overall_accuracy = sum(1 for p in resolved if p.get("prediction_correct")) / len(resolved) * 100
            traded_preds = [p for p in resolved if p.get("was_traded")]
            total_pnl = sum(p.get("trade_pnl", 0) or 0 for p in traded_preds)
        else:
            overall_accuracy = 0
            total_pnl = 0

    finally:
        await datastore.close()

    return {
        "total_predictions": total_predictions,
        "resolved_count": resolved_count,
        "pending_count": len(unresolved),
        "overall_accuracy": overall_accuracy,
        "total_pnl": total_pnl,
        "calibration": calibration,
        "edge_buckets": edge_buckets,
        "city_performance": city_performance,
        "recent_predictions": recent_predictions,
        "timestamp": datetime.now().isoformat(),
    }


async def get_live_positions_data() -> dict[str, Any]:
    """Get live position data from Polymarket."""
    positions = []
    total_value = 0
    total_cost = 0

    try:
        client = PolymarketClient()
        raw_positions = await client.get_positions()

        for pos in raw_positions:
            size = float(pos.get("size", 0) or 0)
            avg_price = float(pos.get("avgPrice", 0) or 0)
            cur_price = float(pos.get("price", 0) or pos.get("curPrice", 0) or avg_price)

            if size <= 0:
                continue

            cost = size * avg_price
            value = size * cur_price
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0

            positions.append({
                "token_id": pos.get("asset", "")[:16] + "...",
                "outcome": pos.get("title", pos.get("outcome", "Unknown")),
                "size": size,
                "avg_price": avg_price,
                "cur_price": cur_price,
                "cost": cost,
                "value": value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

            total_value += value
            total_cost += cost

    except Exception as e:
        pass  # Will return empty list

    return {
        "positions": positions,
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_value - total_cost,
        "position_count": len(positions),
    }


# =============================================================================
# Web Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    data = await get_dashboard_data()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, **data}
    )


@app.get("/positions", response_class=HTMLResponse)
async def positions_page(request: Request):
    """Positions detail page."""
    data = await get_dashboard_data()
    return templates.TemplateResponse(
        "positions.html",
        {"request": request, **data}
    )


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    """Trades history page."""
    data = await get_dashboard_data()
    return templates.TemplateResponse(
        "trades.html",
        {"request": request, **data}
    )


@app.get("/opportunities", response_class=HTMLResponse)
async def opportunities_page(request: Request):
    """Opportunities page."""
    data = await get_dashboard_data()
    return templates.TemplateResponse(
        "opportunities.html",
        {"request": request, **data}
    )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """ML Analytics page."""
    data = await get_dashboard_data()
    ml_data = await get_ml_analytics_data()
    return templates.TemplateResponse(
        "analytics.html",
        {"request": request, **data, **ml_data}
    )


@app.get("/live", response_class=HTMLResponse)
async def live_positions_page(request: Request):
    """Live Polymarket positions page."""
    data = await get_dashboard_data()
    live_data = await get_live_positions_data()
    return templates.TemplateResponse(
        "live.html",
        {"request": request, **data, **live_data}
    )


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/status")
async def api_status():
    """Get current status as JSON."""
    data = await get_dashboard_data()
    return JSONResponse(data)


@app.get("/api/positions")
async def api_positions():
    """Get open positions."""
    datastore = DataStore()
    await datastore.connect()
    try:
        positions = await datastore.get_open_positions()
    finally:
        await datastore.close()
    return JSONResponse({"positions": positions})


@app.get("/api/trades")
async def api_trades():
    """Get recent trades."""
    datastore = DataStore()
    await datastore.connect()
    try:
        trades = await datastore.get_trades()
    finally:
        await datastore.close()
    return JSONResponse({"trades": trades[:50]})


@app.get("/api/opportunities")
async def api_opportunities():
    """Get recent opportunities."""
    datastore = DataStore()
    await datastore.connect()
    try:
        opps = await datastore.get_opportunities(min_edge=0.05)
    finally:
        await datastore.close()
    return JSONResponse({"opportunities": opps[:50]})


@app.get("/api/analytics")
async def api_analytics():
    """Get ML analytics data."""
    data = await get_ml_analytics_data()
    return JSONResponse(data)


@app.get("/api/live-positions")
async def api_live_positions():
    """Get live Polymarket positions."""
    data = await get_live_positions_data()
    return JSONResponse(data)


@app.get("/api/predictions")
async def api_predictions(resolved: bool = False, traded: bool = False):
    """Get prediction records."""
    datastore = DataStore()
    await datastore.connect()
    try:
        if resolved:
            preds = await datastore.get_resolved_predictions(traded_only=traded)
        else:
            preds = await datastore.get_unresolved_predictions()
    finally:
        await datastore.close()
    return JSONResponse({"predictions": preds[:100]})


@app.post("/api/bot/start")
async def start_bot(background_tasks: BackgroundTasks, mode: str = "paper", mock: bool = True):
    """Start the trading bot."""
    global _bot, _bot_task

    if _bot and _bot._running:
        return JSONResponse({"error": "Bot already running"}, status_code=400)

    trading_mode = TradingMode(mode)
    _bot = TradingBot(mode=trading_mode, use_mock=mock)

    async def run_bot():
        await _bot.start()

    _bot_task = asyncio.create_task(run_bot())

    return JSONResponse({"status": "started", "mode": mode, "mock": mock})


@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot."""
    global _bot, _bot_task

    if not _bot:
        return JSONResponse({"error": "Bot not running"}, status_code=400)

    await _bot.stop()
    if _bot_task:
        _bot_task.cancel()

    _bot = None
    _bot_task = None

    return JSONResponse({"status": "stopped"})


# =============================================================================
# Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
