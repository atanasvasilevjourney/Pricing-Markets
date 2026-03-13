"""
Signal classification & trade decision logic.

Converts three-source edge (ML + LLM + LMSR) into actionable signals.
Uses impact-adjusted Kelly sizing when LMSR microstructure data is
available; falls back to naive Kelly otherwise.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Optional, Dict

from .kelly import KellyCriterion
from .lmsr_features import LMSRAdapter


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
    direction: str
    position_size: float
    composite_edge: float
    confidence: float
    p_true: float
    rationale: str
    lmsr_strength: str = "none"
    impact_bps: float = 0.0
    b_calibrated: float = 0.0
    capacity_usd: float = 0.0


class SignalClassifier:
    """
    Signal thresholds (calibrated for three-source fusion):
      STRONG: edge ≥ 0.12, confidence ≥ 0.70, LMSR ≠ weak
      BUY:    edge ≥ 0.08, confidence ≥ 0.55

    The LMSR strength acts as a *veto* on STRONG signals:
    if LMSR says "weak" we downgrade to BUY even if edge is large,
    because thin microstructure means execution risk.
    """

    THRESHOLDS = {
        "strong": {"edge": 0.12, "confidence": 0.70},
        "buy": {"edge": 0.08, "confidence": 0.55},
    }

    def __init__(self, lmsr_adapter: Optional[LMSRAdapter] = None):
        self.lmsr = lmsr_adapter or LMSRAdapter()

    def classify(
        self,
        opportunity: dict,
        bankroll: float,
    ) -> TradeDecision:
        edge = opportunity["composite_edge"]
        direction = opportunity["direction"]
        confidence = opportunity["confidence"]
        current = opportunity["current_price"]

        p_true = (
            opportunity.get("llm_fair_value")
            or opportunity.get("ml_prediction")
            or current
        )

        lmsr_data = opportunity.get("lmsr_analysis") or {}
        lmsr_strength = lmsr_data.get("lmsr_strength", "none")
        b_calibrated = lmsr_data.get("b_calibrated", 0.0)
        capacity_usd = lmsr_data.get("capacity_usd", 0.0)
        impact_info = lmsr_data.get("impact_at_ref_size") or {}
        impact_bps = impact_info.get("slippage_bps", 0.0)

        # ── Signal classification ────────────────────────────────────
        signal = self._classify_signal(
            edge, direction, confidence, lmsr_strength,
        )

        # ── Position sizing ──────────────────────────────────────────
        if signal == Signal.HOLD:
            size = 0.0
        elif b_calibrated > 0:
            impact_fn = partial(
                self.lmsr.impact_cost, current, b_calibrated,
            )
            size = KellyCriterion.position_size_impact_adjusted(
                p_true=p_true,
                p_market=current,
                confidence=confidence,
                bankroll=bankroll,
                impact_cost_fn=impact_fn,
            )
        else:
            size = KellyCriterion.position_size(
                p_true=p_true,
                p_market=current,
                confidence=confidence,
                bankroll=bankroll,
            )

        if capacity_usd > 0:
            size = min(size, capacity_usd)

        return TradeDecision(
            signal=signal,
            market_id=opportunity["market_id"],
            question=opportunity["question"],
            current_price=current,
            direction=direction,
            position_size=round(size, 2),
            composite_edge=edge,
            confidence=confidence,
            p_true=p_true,
            rationale=(
                f"ML edge {opportunity['ml_edge']:.3f}, "
                f"LLM edge {opportunity.get('llm_edge', 0):.3f}, "
                f"LMSR edge {opportunity.get('lmsr_edge', 0):.3f} "
                f"[{lmsr_strength}], "
                f"impact {impact_bps:.0f}bps, "
                f"consensus {direction}"
            ),
            lmsr_strength=lmsr_strength,
            impact_bps=impact_bps,
            b_calibrated=b_calibrated,
            capacity_usd=capacity_usd,
        )

    @staticmethod
    def _classify_signal(
        edge: float,
        direction: str,
        confidence: float,
        lmsr_strength: str,
    ) -> Signal:
        strong_t = SignalClassifier.THRESHOLDS["strong"]
        buy_t = SignalClassifier.THRESHOLDS["buy"]

        if edge < buy_t["edge"] or confidence < buy_t["confidence"]:
            return Signal.HOLD

        is_strong = (
            edge >= strong_t["edge"]
            and confidence >= strong_t["confidence"]
            and lmsr_strength != "weak"
        )

        if direction == "YES":
            return Signal.STRONG_BUY_YES if is_strong else Signal.BUY_YES
        else:
            return Signal.STRONG_BUY_NO if is_strong else Signal.BUY_NO
