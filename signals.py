"""
Signal classification & trade decision logic.
Converts edge + confidence into actionable signals.
"""

from dataclasses import dataclass
from enum import Enum

from .kelly import KellyCriterion


class Signal(Enum):
    STRONG_BUY_YES = "STRONG_BUY_YES"
    BUY_YES = "BUY_YES"
    HOLD = "HOLD"
    BUY_NO = "BUY_NO"
    STRONG_BUY_NO = "STRONG_BUY_NO"


@dataclass
class TradeDecision:
    signal: Signal
    market_id: str
    question: str
    current_price: float
    direction: str          # "YES" or "NO"
    position_size: float    # In USD
    composite_edge: float
    confidence: float
    p_true: float
    rationale: str


class SignalClassifier:
    """
    Typical distribution per 30 markets:
    ~6 STRONG_BUY, ~8 BUY, ~9 NO, ~2 HOLD
    """

    THRESHOLDS = {
        "strong": {"edge": 0.15, "confidence": 0.70},
        "buy": {"edge": 0.08, "confidence": 0.55},
    }

    def classify(self, opportunity: dict, bankroll: float) -> TradeDecision:
        edge = opportunity["composite_edge"]
        direction = opportunity["direction"]
        confidence = opportunity["confidence"]
        current = opportunity["current_price"]
        p_true = (
            opportunity.get("llm_fair_value")
            or opportunity.get("ml_prediction")
            or current
        )

        if edge < 0.02:
            signal = Signal.HOLD
        elif direction == "YES":
            if (
                edge >= self.THRESHOLDS["strong"]["edge"]
                and confidence >= self.THRESHOLDS["strong"]["confidence"]
            ):
                signal = Signal.STRONG_BUY_YES
            elif (
                edge >= self.THRESHOLDS["buy"]["edge"]
                and confidence >= self.THRESHOLDS["buy"]["confidence"]
            ):
                signal = Signal.BUY_YES
            else:
                signal = Signal.HOLD
        else:
            if (
                edge >= self.THRESHOLDS["strong"]["edge"]
                and confidence >= self.THRESHOLDS["strong"]["confidence"]
            ):
                signal = Signal.STRONG_BUY_NO
            elif (
                edge >= self.THRESHOLDS["buy"]["edge"]
                and confidence >= self.THRESHOLDS["buy"]["confidence"]
            ):
                signal = Signal.BUY_NO
            else:
                signal = Signal.HOLD

        size = (
            KellyCriterion.position_size(
                p_true=p_true,
                p_market=current,
                confidence=confidence,
                bankroll=bankroll,
            )
            if signal != Signal.HOLD
            else 0.0
        )

        return TradeDecision(
            signal=signal,
            market_id=opportunity["market_id"],
            question=opportunity["question"],
            current_price=current,
            direction=direction,
            position_size=size,
            composite_edge=edge,
            confidence=confidence,
            p_true=p_true,
            rationale=(
                f"ML edge {opportunity['ml_edge']:.3f}, "
                f"LLM edge {opportunity['llm_edge']:.3f}, "
                f"consensus direction {direction}"
            ),
        )
