"""
Risk management — hard rules and position monitoring.
All gates must pass before a trade executes.
"""

import time
import logging
from typing import Tuple, List

from .config import (
    MAX_SINGLE_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_LIQUIDITY,
    MAX_CORRELATED_CRYPTO,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    TIME_DECAY_DAYS,
)
from .signals import Signal, TradeDecision

log = logging.getLogger(__name__)

CRYPTO_KEYWORDS = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "xrp", "cardano", "ada", "dogecoin", "doge",
})


class RiskGate:
    """Hard risk rules — all must pass before a trade executes."""

    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        self.open_positions: List[dict] = []

    def check_all(self, decision: TradeDecision) -> Tuple[bool, str]:
        checks = [
            self._check_max_single_position,
            self._check_max_total_exposure,
            self._check_min_liquidity,
            self._check_signal_strength,
            self._check_correlation,
        ]
        for check_fn in checks:
            approved, reason = check_fn(decision)
            if not approved:
                return False, reason
        return True, "All checks passed"

    def _check_max_single_position(self, d: TradeDecision) -> Tuple[bool, str]:
        max_allowed = self.bankroll * MAX_SINGLE_POSITION_PCT
        if d.position_size > max_allowed:
            return (
                False,
                f"Position ${d.position_size:.2f} exceeds "
                f"{MAX_SINGLE_POSITION_PCT:.0%} cap ${max_allowed:.2f}",
            )
        return True, "OK"

    def _check_max_total_exposure(self, d: TradeDecision) -> Tuple[bool, str]:
        total_open = sum(p.get("size", 0) for p in self.open_positions)
        max_total = self.bankroll * MAX_TOTAL_EXPOSURE_PCT
        if (total_open + d.position_size) > max_total:
            return (
                False,
                f"Total exposure ${total_open + d.position_size:.2f} "
                f"exceeds {MAX_TOTAL_EXPOSURE_PCT:.0%} cap",
            )
        return True, "OK"

    def _check_min_liquidity(self, d: TradeDecision) -> Tuple[bool, str]:
        liq = getattr(d, "liquidity", MIN_LIQUIDITY + 1)
        if liq < MIN_LIQUIDITY:
            return False, f"Liquidity ${liq:.0f} below ${MIN_LIQUIDITY:,} minimum"
        return True, "OK"

    def _check_signal_strength(self, d: TradeDecision) -> Tuple[bool, str]:
        if d.signal == Signal.HOLD:
            return False, "Signal is HOLD"
        return True, "OK"

    def _check_correlation(self, d: TradeDecision) -> Tuple[bool, str]:
        question_words = set(d.question.lower().split())
        is_crypto = bool(question_words & CRYPTO_KEYWORDS)
        if is_crypto:
            existing_crypto = [
                p for p in self.open_positions if p.get("is_crypto", False)
            ]
            if len(existing_crypto) >= MAX_CORRELATED_CRYPTO:
                return False, "Too many correlated crypto positions"
        return True, "OK"

    def register_position(self, position: dict):
        self.open_positions.append(position)

    def remove_position(self, market_id: str):
        self.open_positions = [
            p for p in self.open_positions if p.get("market_id") != market_id
        ]


class PositionMonitor:
    """
    Monitors open positions and triggers exits.
    Henry strategy: EXIT when market reprices, NOT at resolution.
    """

    def should_exit(
        self, position: dict, current_price: float
    ) -> Tuple[bool, str]:
        entry_price = position["entry_price"]
        direction = position["direction"]

        if direction == "YES":
            gain = current_price - entry_price
            loss = entry_price - current_price
        else:
            gain = entry_price - current_price
            loss = current_price - entry_price

        if gain >= entry_price * TAKE_PROFIT_PCT:
            return True, f"Take profit: +{gain / entry_price * 100:.1f}%"

        if loss >= entry_price * STOP_LOSS_PCT:
            return True, f"Stop loss: -{loss / entry_price * 100:.1f}%"

        days_held = (time.time() - position["entry_time"]) / 86400
        if days_held > TIME_DECAY_DAYS:
            return True, f"Time exit: held {days_held:.1f} days"

        return False, "Hold"
