"""
Core mispricing detection — combines ML edge and LLM edge.
Formula: composite_edge = ml_edge * 0.4 + llm_edge * 0.6
"""

import logging
from typing import Optional, List

from .config import ML_WEIGHT, LLM_WEIGHT, MIN_EDGE_THRESHOLD
from .features import FeatureExtractor
from .ml_engine import MLPredictionEngine
from .llm_engine import LLMProbabilityEngine
from .fetcher import PolymarketFetcher

log = logging.getLogger(__name__)


class MispricingDetector:

    def __init__(
        self,
        ml_engine: MLPredictionEngine,
        llm_engine: LLMProbabilityEngine,
        fetcher: PolymarketFetcher,
    ):
        self.ml = ml_engine
        self.llm = llm_engine
        self.fetcher = fetcher
        self.extractor = FeatureExtractor()

    def scan_all_markets(
        self,
        n_markets: int = 50,
        min_volume: float = 5_000,
        min_edge: float = MIN_EDGE_THRESHOLD,
    ) -> List[dict]:
        """
        Main scan loop. Returns ranked list of mispriced markets
        with edge > min_edge (default 8%).
        """
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
        """Full analysis pipeline for a single market."""
        question = market.get("question", "")
        market_id = market.get("id", "")
        token_id = (market.get("clobTokenIds") or [""])[0]
        prices = self.fetcher.get_best_prices(token_id)
        current_price = prices["mid"]

        if current_price <= 0.01 or current_price >= 0.99:
            return None

        # Step 1: ML-based edge
        features = self.extractor.extract(market, self.fetcher)
        ml_result = self.ml.predict(features, current_price)
        ml_edge = ml_result["edge"]
        ml_direction = ml_result["direction"]

        # Step 2: LLM-based edge (only if ML found > 4% edge — saves API cost)
        llm_result = None
        if abs(ml_edge) > 0.04:
            llm_result = self.llm.estimate_probability(
                question=question,
                current_price=current_price,
                market_metadata=market,
            )

        # Step 3: Composite score
        if llm_result:
            llm_edge = abs(llm_result["fair_value"] - current_price)
            llm_direction = (
                "YES" if llm_result["fair_value"] > current_price else "NO"
            )
            direction_agreement = ml_direction == llm_direction
            composite_edge = (
                (ml_edge * ML_WEIGHT + llm_edge * LLM_WEIGHT)
                if direction_agreement
                else min(ml_edge, llm_edge) * 0.3
            )
        else:
            composite_edge = ml_edge * ML_WEIGHT
            llm_direction = ml_direction
            llm_edge = 0.0

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
            "composite_edge": composite_edge,
            "direction": ml_direction,
            "confidence": ml_result.get("confidence", 0),
            "volume_24h": float(market.get("volume24hr", 0) or 0),
            "liquidity": float(market.get("liquidityClob", 0) or 0),
            "market": market,
        }


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
