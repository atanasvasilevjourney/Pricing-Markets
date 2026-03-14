"""
Kill switch — daily loss limit and max trades per day.

Persists state in logs/daily_state.json so that after a restart the bot
still respects the same daily caps. When either limit is hit, the bot
exits cleanly (no new orders until the next calendar day).

Usage:
    Before each scan cycle: should_stop, reason = check_kill_switch(bankroll)
    After placing orders: record_trades(n_placed, current_balance=None)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from .config import (
    DAILY_LOSS_LIMIT_PCT,
    MAX_TRADES_PER_DAY,
    BALANCE_FILE,
)

log = logging.getLogger(__name__)

STATE_PATH = Path("logs") / "daily_state.json"


def _ensure_logs() -> None:
    Path("logs").mkdir(exist_ok=True)


def _load_state() -> dict:
    _ensure_logs()
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not load daily state: %s", e)
        return {}


def _save_state(state: dict) -> None:
    _ensure_logs()
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        log.warning("Could not save daily state: %s", e)


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def get_current_balance(bankroll_fallback: float) -> float:
    """Read current balance from BALANCE_FILE if set, else use bankroll."""
    if not BALANCE_FILE:
        return bankroll_fallback
    path = Path(BALANCE_FILE)
    if not path.exists():
        return bankroll_fallback
    try:
        text = path.read_text(encoding="utf-8").strip()
        return float(text)
    except (ValueError, OSError):
        return bankroll_fallback


def check_kill_switch(bankroll: float) -> Tuple[bool, str]:
    """
    Call before each scan cycle. Returns (should_stop, reason).
    If should_stop is True, exit the main loop and do not place more orders today.
    """
    today = _today()
    state = _load_state()

    # New day: reset daily counters
    if state.get("date") != today:
        current = get_current_balance(bankroll)
        state = {
            "date": today,
            "start_balance": current,
            "trades_count": 0,
        }
        _save_state(state)
        return False, ""

    # Check max trades per day
    count = state.get("trades_count", 0)
    if count >= MAX_TRADES_PER_DAY:
        return True, (
            f"Kill switch: max trades per day reached ({count}/{MAX_TRADES_PER_DAY}). "
            "Stopping until next calendar day."
        )

    # Check daily loss limit
    start_balance = state.get("start_balance")
    if start_balance is not None and start_balance > 0:
        current = get_current_balance(bankroll)
        loss_pct = (start_balance - current) / start_balance
        if loss_pct >= DAILY_LOSS_LIMIT_PCT:
            return True, (
                f"Kill switch: daily loss limit reached "
                f"({loss_pct:.1%} >= {DAILY_LOSS_LIMIT_PCT:.1%}). "
                f"Start=${start_balance:.2f} Current=${current:.2f}. "
                "Stopping until next calendar day."
            )

    return False, ""


def record_trades(n_placed: int, current_balance: float | None = None) -> None:
    """
    Call after placing orders. Updates daily trade count and optionally
    the stored balance (if your execution layer writes BALANCE_FILE).
    """
    if n_placed <= 0:
        return

    today = _today()
    state = _load_state()

    if state.get("date") != today:
        state = {
            "date": today,
            "start_balance": current_balance or get_current_balance(500.0),
            "trades_count": 0,
        }

    state["trades_count"] = state.get("trades_count", 0) + n_placed
    if current_balance is not None:
        state["current_balance"] = current_balance

    _save_state(state)
    log.info(
        "Daily state: %d trades placed today (cap %d)",
        state["trades_count"],
        MAX_TRADES_PER_DAY,
    )
