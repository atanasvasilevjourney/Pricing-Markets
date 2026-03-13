"""
CSV logger for paper-trading and live-trading sessions.

Appends one row per signal/decision to a daily CSV under logs/.
Designed to be wired into the main scan cycle with zero friction.

Columns logged:
    timestamp, market_id, question, signal, direction, current_price,
    p_true, composite_edge, ml_edge, llm_edge, lmsr_edge,
    lmsr_strength, b_calibrated, impact_bps, capacity_usd,
    position_size, confidence, rationale

Usage:
    logger = PaperTradeLogger()
    logger.log_decision(decision, opportunity)
    logger.log_arb(arb_opp)
    logger.flush()     # force write (auto-flushes every row by default)
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .signals import TradeDecision

log = logging.getLogger(__name__)

TRADE_FIELDS = [
    "timestamp",
    "market_id",
    "question",
    "signal",
    "direction",
    "current_price",
    "p_true",
    "composite_edge",
    "ml_edge",
    "llm_edge",
    "lmsr_edge",
    "lmsr_strength",
    "b_calibrated",
    "impact_bps",
    "capacity_usd",
    "position_size",
    "confidence",
    "rationale",
]

ARB_FIELDS = [
    "timestamp",
    "type",
    "market_id",
    "question",
    "total_cost",
    "profit_pct",
    "action",
]


class PaperTradeLogger:
    """Append-only CSV logger for trade signals and arb opportunities."""

    def __init__(self, log_dir: str = "logs"):
        self._dir = Path(log_dir)
        self._dir.mkdir(exist_ok=True)
        self._date_str = datetime.now().strftime("%Y%m%d")

        self._trade_path = self._dir / f"paper_trades_{self._date_str}.csv"
        self._arb_path = self._dir / f"paper_arbs_{self._date_str}.csv"

        self._ensure_header(self._trade_path, TRADE_FIELDS)
        self._ensure_header(self._arb_path, ARB_FIELDS)

        self._trade_count = 0
        self._arb_count = 0

    @staticmethod
    def _ensure_header(path: Path, fields: list):
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

    def log_decision(
        self,
        decision: TradeDecision,
        opportunity: Optional[dict] = None,
    ):
        """Log one trade decision (signal) to the daily CSV."""
        opp = opportunity or {}
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "market_id": decision.market_id,
            "question": decision.question[:120],
            "signal": decision.signal.value,
            "direction": decision.direction,
            "current_price": f"{decision.current_price:.4f}",
            "p_true": f"{decision.p_true:.4f}",
            "composite_edge": f"{decision.composite_edge:.4f}",
            "ml_edge": f"{opp.get('ml_edge', 0):.4f}",
            "llm_edge": f"{opp.get('llm_edge', 0):.4f}",
            "lmsr_edge": f"{opp.get('lmsr_edge', 0):.4f}",
            "lmsr_strength": decision.lmsr_strength,
            "b_calibrated": f"{decision.b_calibrated:.1f}",
            "impact_bps": f"{decision.impact_bps:.0f}",
            "capacity_usd": f"{decision.capacity_usd:.2f}",
            "position_size": f"{decision.position_size:.2f}",
            "confidence": f"{decision.confidence:.4f}",
            "rationale": decision.rationale,
        }
        self._append_row(self._trade_path, TRADE_FIELDS, row)
        self._trade_count += 1

    def log_arb(self, arb: dict):
        """Log one arbitrage opportunity to the daily CSV."""
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": arb.get("type", ""),
            "market_id": arb.get("market_id", arb.get("event_slug", "")),
            "question": str(arb.get("question", ""))[:120],
            "total_cost": f"{arb.get('total_cost', 0):.4f}",
            "profit_pct": f"{arb.get('profit_pct', 0):.2f}",
            "action": arb.get("action", ""),
        }
        self._append_row(self._arb_path, ARB_FIELDS, row)
        self._arb_count += 1

    def summary(self) -> str:
        return (
            f"Paper logger: {self._trade_count} trades, "
            f"{self._arb_count} arbs logged to {self._dir}/"
        )

    @staticmethod
    def _append_row(path: Path, fields: list, row: dict):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writerow(row)
