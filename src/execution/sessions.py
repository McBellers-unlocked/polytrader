"""Session management for tracking trading periods separately."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


@dataclass
class TradingSession:
    """Represents a trading session/period."""

    session_id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    notes: str = ""

    # Strategy parameters at session start
    min_edge_threshold: float = 0.0
    min_confidence: float = 0.0
    max_positions: int = 0

    # Performance metrics (updated on close)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    total_cost: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def return_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return self.total_pnl / self.total_cost

    @property
    def duration_days(self) -> float:
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds() / 86400

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradingSession":
        d["started_at"] = datetime.fromisoformat(d["started_at"])
        if d["ended_at"]:
            d["ended_at"] = datetime.fromisoformat(d["ended_at"])
        return cls(**d)


@dataclass
class SessionManager:
    """Manages trading sessions and archives."""

    sessions_file: Path = field(default_factory=lambda: Path("sessions.json"))
    _sessions: list[TradingSession] = field(default_factory=list)
    _current_session: Optional[TradingSession] = None

    def __post_init__(self):
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from file."""
        if self.sessions_file.exists():
            try:
                data = json.loads(self.sessions_file.read_text())
                self._sessions = [TradingSession.from_dict(s) for s in data.get("sessions", [])]

                # Find active session
                for session in self._sessions:
                    if session.is_active:
                        self._current_session = session
                        break
            except Exception as e:
                print(f"Warning: Could not load sessions: {e}")
                self._sessions = []

    def _save_sessions(self) -> None:
        """Save sessions to file."""
        data = {
            "sessions": [s.to_dict() for s in self._sessions],
            "updated_at": datetime.now().isoformat(),
        }
        self.sessions_file.write_text(json.dumps(data, indent=2))

    @property
    def current_session(self) -> Optional[TradingSession]:
        return self._current_session

    @property
    def all_sessions(self) -> list[TradingSession]:
        return self._sessions

    @property
    def archived_sessions(self) -> list[TradingSession]:
        return [s for s in self._sessions if not s.is_active]

    def start_new_session(
        self,
        name: str,
        starting_bankroll: float,
        min_edge_threshold: float = 0.15,
        min_confidence: float = 0.50,
        max_positions: int = 15,
        notes: str = "",
    ) -> TradingSession:
        """
        Start a new trading session.

        Closes any active session first.
        """
        # Close current session if active
        if self._current_session and self._current_session.is_active:
            self.close_current_session(notes="Auto-closed for new session")

        # Create new session
        session = TradingSession(
            session_id=str(uuid.uuid4())[:8],
            name=name,
            started_at=datetime.now(),
            starting_bankroll=starting_bankroll,
            min_edge_threshold=min_edge_threshold,
            min_confidence=min_confidence,
            max_positions=max_positions,
            notes=notes,
        )

        self._sessions.append(session)
        self._current_session = session
        self._save_sessions()

        return session

    def close_current_session(
        self,
        ending_bankroll: float = 0.0,
        total_trades: int = 0,
        winning_trades: int = 0,
        total_pnl: float = 0.0,
        total_cost: float = 0.0,
        notes: str = "",
    ) -> Optional[TradingSession]:
        """Close the current session and archive it."""
        if not self._current_session:
            return None

        self._current_session.ended_at = datetime.now()
        self._current_session.ending_bankroll = ending_bankroll
        self._current_session.total_trades = total_trades
        self._current_session.winning_trades = winning_trades
        self._current_session.total_pnl = total_pnl
        self._current_session.total_cost = total_cost
        if notes:
            self._current_session.notes = notes

        closed_session = self._current_session
        self._current_session = None
        self._save_sessions()

        return closed_session

    def update_current_session(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        total_cost: float,
    ) -> None:
        """Update metrics for current session."""
        if self._current_session:
            self._current_session.total_trades = total_trades
            self._current_session.winning_trades = winning_trades
            self._current_session.total_pnl = total_pnl
            self._current_session.total_cost = total_cost
            self._save_sessions()

    def get_session_by_id(self, session_id: str) -> Optional[TradingSession]:
        """Get a session by its ID."""
        for session in self._sessions:
            if session.session_id == session_id:
                return session
        return None

    def compare_sessions(self) -> str:
        """Generate a comparison report of all sessions."""
        if not self._sessions:
            return "No sessions found."

        lines = [
            "=" * 80,
            "SESSION COMPARISON",
            "=" * 80,
            "",
        ]

        for session in self._sessions:
            status = "🟢 ACTIVE" if session.is_active else "📦 ARCHIVED"
            pnl_icon = "🟢" if session.total_pnl >= 0 else "🔴"

            lines.append(f"{status} Session: {session.name} ({session.session_id})")
            lines.append(f"   Started: {session.started_at.strftime('%Y-%m-%d %H:%M')}")
            if session.ended_at:
                lines.append(f"   Ended:   {session.ended_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"   Duration: {session.duration_days:.1f} days")
            lines.append(f"   Settings: edge≥{session.min_edge_threshold:.0%}, conf≥{session.min_confidence:.0%}, max={session.max_positions} pos")
            lines.append(f"   {pnl_icon} Performance: {session.total_trades} trades, {session.win_rate:.1%} win rate, ${session.total_pnl:+.2f} ({session.return_pct:.1%})")
            lines.append("")

        return "\n".join(lines)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the session manager (for testing)."""
    global _session_manager
    _session_manager = None
