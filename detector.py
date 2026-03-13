"""
Three-source mispricing detection: ML + LLM + LMSR microstructure.

Signal fusion formula:
  composite_edge = w_ml·e_ml + w_llm·e_llm + w_lmsr·e_lmsr
                                                  (when all agree)

When directions disagree the composite is penalised:
  - Two agree, one disagrees → weight of dissenter zeroed, rest renormalised.
  - All three disagree → edge = min(edges) × 0.2  (near-zero conviction).

The LMSR layer also provides:
  - Calibrated b (market depth proxy)
  - Impact cost at the reference order size
  - Capacity ceiling for the market
"""

import logging
from typing import Optional, List

from .config import (
    ML_WEIGHT,
    LLM_WEIGHT,
    LMSR_WEIGHT,
    MIN_EDGE_THRESHOLD,
)
from .features import FeatureExtractor
from .ml_engine import MLPredictionEngine
from .llm_engine import LLMProbabilityEngine
from .lmsr_features import LMSRAdapter
from .fetcher import PolymarketFetcher

log = logging.getLogger(__name__)


class MispricingDetector:

    def __init__(
        self,
        ml_engine: MLPredictionEngine,
        llm_engine: LLMProbabilityEngine,
        fetcher: PolymarketFetcher,
        lmsr_adapter: Optional[LMSRAdapter] = None,
    ):
        self.ml = ml_engine
        self.llm = llm_engine
        self.fetcher = fetcher
        self.lmsr = lmsr_adapter or LMSRAdapter()
        self.extractor = FeatureExtractor()

    def scan_all_markets(
        self,
        n_markets: int = 50,
        min_volume: float = 5_000,
        min_edge: float = MIN_EDGE_THRESHOLD,
    ) -> List[dict]:
        markets = self.fetcher.get_markets(
            limit=n_markets * 3,
            order="volume24hr",
            min_volume=min_volume,
        )

        opportunities: list = []
        for market in markets[:n_markets]:
            try:
                opp = self.analyze_market(market)
                if opp and opp["composite_edge"] >= min_edge:
                    opportunities.append(opp)
            except Exception as e:
                log.debug(
                    "Error analyzing %s: %s",
                    market.get("question", "?"), e,
                )

        opportunities.sort(key=lambda x: x["composite_edge"], reverse=True)
        return opportunities

    def analyze_market(self, market: dict) -> Optional[dict]:
        """Full three-source analysis pipeline for a single market."""
        question = market.get("question", "")
        market_id = market.get("id", "")
        token_id = (market.get("clobTokenIds") or [""])[0]
        prices = self.fetcher.get_best_prices(token_id)
        current_price = prices["mid"]

        if current_price <= 0.01 or current_price >= 0.99:
            return None

        # ── Source 1: ML ──────────────────────────────────────────────
        features = self.extractor.extract(market, self.fetcher)
        ml_result = self.ml.predict(features, current_price)
        ml_edge = ml_result["edge"]
        ml_direction = ml_result["direction"]

        # ── Source 2: LLM (gated on ML edge > 4%) ────────────────────
        llm_result = None
        if abs(ml_edge) > 0.04:
            llm_result = self.llm.estimate_probability(
                question=question,
                current_price=current_price,
                market_metadata=market,
            )

        if llm_result:
            llm_edge = abs(llm_result["fair_value"] - current_price)
            llm_direction = (
                "YES" if llm_result["fair_value"] > current_price else "NO"
            )
        else:
            llm_edge = 0.0
            llm_direction = ml_direction

        # ── Source 3: LMSR microstructure ─────────────────────────────
        p_model = _best_model_price(ml_result, llm_result, current_price)
        model_confidence = ml_result.get("confidence", 0.5)

        lmsr_analysis = self.lmsr.analyze_market(
            market, self.fetcher, p_model, model_confidence,
        )
        lmsr_edge = lmsr_analysis["lmsr_edge"]
        lmsr_direction = (
            "YES" if lmsr_analysis["lmsr_side"] == "BUY" else "NO"
        )

        # ── Three-source fusion ───────────────────────────────────────
        composite_edge, direction = _fuse_edges(
            ml_edge, ml_direction,
            llm_edge, llm_direction,
            lmsr_edge, lmsr_direction,
        )

        return {
            "market_id": market_id,
            "question": question,
            "current_price": current_price,
            "ml_prediction": ml_result.get("predicted_price"),
            "ml_edge": ml_edge,
            "llm_fair_value": (
                llm_result.get("fair_value") if llm_result else None
            ),
            "llm_edge": llm_edge,
            "lmsr_edge": lmsr_edge,
            "lmsr_analysis": lmsr_analysis,
            "composite_edge": composite_edge,
            "direction": direction,
            "confidence": ml_result.get("confidence", 0),
            "volume_24h": float(market.get("volume24hr", 0) or 0),
            "liquidity": float(market.get("liquidityClob", 0) or 0),
            "market": market,
        }


# ── Fusion helpers ────────────────────────────────────────────────────

def _best_model_price(
    ml_result: dict,
    llm_result: Optional[dict],
    current_price: float,
) -> float:
    """Weighted average of ML predicted price and LLM fair value."""
    ml_p = ml_result.get("predicted_price", current_price)
    if llm_result and llm_result.get("fair_value"):
        llm_p = llm_result["fair_value"]
        return ml_p * (ML_WEIGHT / (ML_WEIGHT + LLM_WEIGHT)) + \
               llm_p * (LLM_WEIGHT / (ML_WEIGHT + LLM_WEIGHT))
    return ml_p


def _fuse_edges(
    ml_edge: float, ml_dir: str,
    llm_edge: float, llm_dir: str,
    lmsr_edge: float, lmsr_dir: str,
) -> tuple:
    """
    Fuse three edge sources with direction-agreement logic.

    Returns (composite_edge, consensus_direction).
    """
    sources = [
        (ML_WEIGHT, ml_edge, ml_dir),
        (LLM_WEIGHT, llm_edge, llm_dir),
        (LMSR_WEIGHT, lmsr_edge, lmsr_dir),
    ]

    active = [(w, e, d) for w, e, d in sources if e > 0]
    if not active:
        return 0.0, "YES"

    direction_votes = {}
    for w, e, d in active:
        direction_votes[d] = direction_votes.get(d, 0) + w

    consensus_dir = max(direction_votes, key=direction_votes.get)

    agreeing = [(w, e) for w, e, d in active if d == consensus_dir]
    disagreeing = [(w, e) for w, e, d in active if d != consensus_dir]

    if not disagreeing:
        total_w = sum(w for w, _ in agreeing)
        composite = sum(w * e for w, e in agreeing) / total_w if total_w > 0 else 0.0
    elif len(agreeing) >= 2:
        total_w = sum(w for w, _ in agreeing)
        composite = sum(w * e for w, e in agreeing) / total_w if total_w > 0 else 0.0
        composite *= 0.85
    else:
        composite = min(e for _, e in active) * 0.20

    return composite, consensus_dir


def henry_filter(opportunity: dict) -> bool:
    """
    Filter for the Henry strategy:
    Buy underpriced low-probability events and sell on repricing.
    """
    p = opportunity["current_price"]
    edge = opportunity["composite_edge"]
    vol = opportunity["volume_24h"]
    liq = opportunity["liquidity"]

    return all([
        0.02 <= p <= 0.25,
        edge >= 0.08,
        vol >= 5_000,
        liq >= 2_000,
        opportunity["direction"] == "YES",
    ])
