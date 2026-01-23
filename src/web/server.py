"""FastAPI web server for Polytrader dashboard.

Provides a web interface to monitor the trading bot.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import get_settings, CITIES, TradingMode
from src.execution.datastore import DataStore
from src.risk.risk_manager import RiskManager
from src.main import TradingBot

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
