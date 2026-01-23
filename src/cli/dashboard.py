"""Rich CLI dashboard for Polytrader trading bot.

Provides beautiful terminal output for:
- Current open positions with unrealized P&L
- Today's trades with realized P&L
- Active opportunities being monitored
- Risk status (daily P&L, weekly P&L, drawdown)
- Next scheduled actions
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich import box

from src.config import get_settings
from src.execution.datastore import DataStore
from src.risk.risk_manager import RiskManager


class Dashboard:
    """Rich terminal dashboard for Polytrader."""

    def __init__(self, db_path: str | None = None):
        """Initialize the dashboard."""
        self.settings = get_settings()
        self.db_path = db_path or self.settings.db_path
        self.console = Console()
        self.datastore = DataStore(db_path=self.db_path)

    async def show_full(self) -> None:
        """Show the full dashboard with all panels."""
        await self.datastore.connect()
        try:
            self.console.clear()
            self._print_header()
            await self._print_risk_status()
            await self._print_positions()
            await self._print_todays_trades()
            await self._print_opportunities()
            self._print_next_actions()
        finally:
            await self.datastore.close()

    async def show_positions(self) -> None:
        """Show only positions panel."""
        await self.datastore.connect()
        try:
            self._print_header()
            await self._print_positions(expanded=True)
        finally:
            await self.datastore.close()

    async def show_pnl(self) -> None:
        """Show P&L summary."""
        await self.datastore.connect()
        try:
            self._print_header()
            await self._print_pnl_summary()
        finally:
            await self.datastore.close()

    async def show_opportunities(self) -> None:
        """Show active opportunities."""
        await self.datastore.connect()
        try:
            self._print_header()
            await self._print_opportunities(expanded=True)
        finally:
            await self.datastore.close()

    def _print_header(self) -> None:
        """Print dashboard header."""
        header = Text()
        header.append("POLYTRADER", style="bold cyan")
        header.append(" | ", style="dim")
        header.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        header.append(" | ", style="dim")
        header.append(f"Mode: {self.settings.trading_mode.value.upper()}", style="yellow")

        self.console.print(Panel(header, box=box.DOUBLE_EDGE))
        self.console.print()

    async def _print_risk_status(self) -> None:
        """Print risk status panel."""
        risk = RiskManager(
            starting_bankroll=Decimal(str(self.settings.starting_bankroll))
        )
        status = risk.get_status()

        # Create risk status table
        table = Table(
            title="Risk Status",
            box=box.ROUNDED,
            show_header=False,
            title_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        # Bankroll
        bankroll = float(status["bankroll"]["current"])
        starting = float(status["bankroll"]["starting"])
        pnl_pct = ((bankroll - starting) / starting) * 100 if starting > 0 else 0

        table.add_row(
            "Bankroll",
            f"${bankroll:,.2f}",
            self._status_indicator(pnl_pct >= 0),
        )

        # Drawdown
        drawdown = float(status["bankroll"]["drawdown_pct"])
        max_dd = self.settings.max_drawdown_pct * 100
        dd_status = "green" if drawdown < max_dd * 0.5 else ("yellow" if drawdown < max_dd * 0.8 else "red")

        table.add_row(
            "Drawdown",
            f"{drawdown:.1f}% / {max_dd:.0f}%",
            Text("●", style=dd_status),
        )

        # Daily P&L
        daily_pnl = float(status["daily"]["total_pnl_pct"])
        daily_limit = self.settings.daily_loss_stop_pct * 100

        table.add_row(
            "Daily P&L",
            f"{daily_pnl:+.2f}%",
            self._status_indicator(daily_pnl > -daily_limit),
        )

        # Trading status
        trading_allowed = status["stops"]["trading_allowed"]
        table.add_row(
            "Trading",
            "ACTIVE" if trading_allowed else "STOPPED",
            Text("●", style="green" if trading_allowed else "red"),
        )

        # Positions
        pos_count = status["positions"]["count"]
        max_pos = status["positions"]["max"]
        table.add_row(
            "Positions",
            f"{pos_count} / {max_pos}",
            self._status_indicator(pos_count < max_pos),
        )

        self.console.print(table)
        self.console.print()

    async def _print_positions(self, expanded: bool = False) -> None:
        """Print open positions table."""
        positions = await self.datastore.get_open_positions()

        table = Table(
            title="Open Positions",
            box=box.ROUNDED,
            title_style="bold green",
        )
        table.add_column("City", style="cyan")
        table.add_column("Outcome")
        table.add_column("Side", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Edge", justify="right")

        if not positions:
            table.add_row(
                Text("No open positions", style="dim"),
                "", "", "", "", "", "", ""
            )
        else:
            for pos in positions:
                entry = pos.get("entry_price", 0)
                current = pos.get("current_price", entry)
                size = pos.get("current_size", 0)

                # Calculate unrealized P&L
                unrealized = (current - entry) * size
                pnl_style = "green" if unrealized >= 0 else "red"

                table.add_row(
                    pos.get("city", "Unknown"),
                    pos.get("outcome", ""),
                    pos.get("side", "BUY") if pos.get("entry_edge", 0) > 0 else "SELL",
                    f"${entry:.3f}",
                    f"${current:.3f}",
                    f"{size:.0f}",
                    Text(f"${unrealized:+.2f}", style=pnl_style),
                    f"{pos.get('entry_edge', 0)*100:.1f}%",
                )

        self.console.print(table)
        self.console.print()

    async def _print_todays_trades(self) -> None:
        """Print today's executed trades."""
        today = date.today()
        trades = await self.datastore.get_trades(
            start_date=today,
            end_date=today,
        )

        table = Table(
            title=f"Today's Trades ({len(trades)} total)",
            box=box.ROUNDED,
            title_style="bold blue",
        )
        table.add_column("Time", style="dim")
        table.add_column("City", style="cyan")
        table.add_column("Outcome")
        table.add_column("Side", justify="center")
        table.add_column("Price", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Mode", justify="center")

        if not trades:
            table.add_row(
                Text("No trades today", style="dim"),
                "", "", "", "", "", "", ""
            )
        else:
            # Show last 10 trades
            for trade in trades[:10]:
                timestamp = trade.get("timestamp", "")
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except ValueError:
                        time_str = timestamp[:8]
                else:
                    time_str = str(timestamp)[:8]

                edge = trade.get("edge", 0)
                edge_style = "green" if edge > 0 else "red"

                side = trade.get("side", "BUY")
                side_style = "green" if side == "BUY" else "red"

                table.add_row(
                    time_str,
                    trade.get("city", "Unknown"),
                    trade.get("outcome", ""),
                    Text(side, style=side_style),
                    f"${trade.get('price', 0):.3f}",
                    f"{trade.get('size', 0):.0f}",
                    Text(f"{edge*100:+.1f}%", style=edge_style),
                    trade.get("trading_mode", "paper"),
                )

        self.console.print(table)
        self.console.print()

    async def _print_opportunities(self, expanded: bool = False) -> None:
        """Print recent opportunities."""
        today = date.today()
        opps = await self.datastore.get_opportunities(
            start_date=today,
            end_date=today,
            min_edge=0.05,  # Only show 5%+ edge
        )

        # Sort by edge (absolute value)
        opps.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

        table = Table(
            title=f"Active Opportunities ({len(opps)} with 5%+ edge)",
            box=box.ROUNDED,
            title_style="bold yellow",
        )
        table.add_column("City", style="cyan")
        table.add_column("Outcome")
        table.add_column("Side", justify="center")
        table.add_column("Fair", justify="right")
        table.add_column("Market", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("METAR", justify="center")
        table.add_column("Traded", justify="center")

        if not opps:
            table.add_row(
                Text("No opportunities detected", style="dim"),
                "", "", "", "", "", "", "", ""
            )
        else:
            # Show top opportunities
            limit = 20 if expanded else 10
            for opp in opps[:limit]:
                edge = opp.get("edge", 0)
                edge_style = "green" if edge > 0 else "red"

                side = opp.get("side", "HOLD")
                side_style = "green" if side == "BUY" else ("red" if side == "SELL" else "dim")

                metar = "✓" if opp.get("metar_boosted") else ""
                metar_style = "cyan" if metar else "dim"

                traded = "✓" if opp.get("was_traded") else ""
                traded_style = "green" if traded else "dim"

                table.add_row(
                    opp.get("city", "Unknown"),
                    opp.get("outcome", ""),
                    Text(side, style=side_style),
                    f"{opp.get('fair_probability', 0)*100:.1f}%",
                    f"{opp.get('market_price', 0)*100:.1f}%",
                    Text(f"{edge*100:+.1f}%", style=edge_style),
                    f"${opp.get('suggested_size', 0):.0f}",
                    Text(metar, style=metar_style),
                    Text(traded, style=traded_style),
                )

        self.console.print(table)
        self.console.print()

    async def _print_pnl_summary(self) -> None:
        """Print detailed P&L summary."""
        # Get daily summaries for the past week
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        summaries = await self.datastore.get_daily_summaries(
            start_date=start_date,
            end_date=end_date,
        )

        # Summary stats table
        stats_table = Table(
            title="P&L Summary (Last 7 Days)",
            box=box.ROUNDED,
            title_style="bold green",
        )
        stats_table.add_column("Date", style="cyan")
        stats_table.add_column("P&L $", justify="right")
        stats_table.add_column("P&L %", justify="right")
        stats_table.add_column("Trades", justify="right")
        stats_table.add_column("Win Rate", justify="right")
        stats_table.add_column("METAR", justify="right")

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0

        if not summaries:
            stats_table.add_row(
                Text("No trading history", style="dim"),
                "", "", "", "", ""
            )
        else:
            for summary in summaries:
                pnl = summary.get("total_pnl", 0)
                pnl_pct = summary.get("total_pnl_pct", 0)
                n_trades = summary.get("n_trades", 0)
                n_wins = summary.get("n_wins", 0)
                win_rate = summary.get("win_rate")

                total_pnl += pnl
                total_trades += n_trades
                total_wins += n_wins

                pnl_style = "green" if pnl >= 0 else "red"
                wr_style = "green" if (win_rate or 0) >= 50 else "yellow"

                stats_table.add_row(
                    summary.get("date", ""),
                    Text(f"${pnl:+.2f}", style=pnl_style),
                    Text(f"{pnl_pct:+.2f}%", style=pnl_style),
                    str(n_trades),
                    Text(f"{win_rate:.0f}%" if win_rate else "-", style=wr_style),
                    str(summary.get("n_metar_trades", 0)),
                )

        self.console.print(stats_table)
        self.console.print()

        # Totals panel
        if summaries:
            overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
            pnl_style = "green" if total_pnl >= 0 else "red"

            totals = Text()
            totals.append("Weekly Total: ", style="bold")
            totals.append(f"${total_pnl:+.2f}", style=pnl_style)
            totals.append(f" | Trades: {total_trades}", style="dim")
            totals.append(f" | Win Rate: {overall_wr:.0f}%", style="dim")

            self.console.print(Panel(totals, title="Summary", box=box.ROUNDED))
            self.console.print()

    def _print_next_actions(self) -> None:
        """Print next scheduled actions."""
        table = Table(
            title="Next Actions",
            box=box.ROUNDED,
            title_style="bold cyan",
            show_header=False,
        )
        table.add_column("Action", style="white")
        table.add_column("Time", justify="right", style="dim")

        # Calculate next iteration time
        now = datetime.now()
        next_scan = now + timedelta(seconds=30 - (now.second % 30))

        table.add_row("Next market scan", next_scan.strftime("%H:%M:%S"))

        # Price snapshot timing (every 5 min)
        next_snapshot = now + timedelta(minutes=5 - (now.minute % 5), seconds=-now.second)
        table.add_row("Next price snapshot", next_snapshot.strftime("%H:%M:%S"))

        # METAR update (every 15 min)
        next_metar = now + timedelta(minutes=15 - (now.minute % 15), seconds=-now.second)
        table.add_row("Next METAR fetch", next_metar.strftime("%H:%M:%S"))

        # Trading window info
        hour = now.hour
        if 10 <= hour < 18:
            trading_ends = now.replace(hour=18, minute=0, second=0)
            remaining = trading_ends - now
            hours = remaining.seconds // 3600
            mins = (remaining.seconds % 3600) // 60
            table.add_row("Trading window closes", f"{hours}h {mins}m remaining")
        elif hour < 10:
            trading_starts = now.replace(hour=10, minute=0, second=0)
            until = trading_starts - now
            hours = until.seconds // 3600
            mins = (until.seconds % 3600) // 60
            table.add_row("Trading window opens", f"in {hours}h {mins}m")
        else:
            table.add_row("Trading window", "CLOSED (opens 10:00 AM)")

        self.console.print(table)
        self.console.print()

    def _status_indicator(self, is_good: bool) -> Text:
        """Return a status indicator."""
        if is_good:
            return Text("●", style="green")
        return Text("●", style="red")


async def demo_dashboard():
    """Demonstrate the dashboard."""
    dashboard = Dashboard()
    await dashboard.show_full()


if __name__ == "__main__":
    asyncio.run(demo_dashboard())
