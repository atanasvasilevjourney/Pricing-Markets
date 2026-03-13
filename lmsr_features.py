"""
LMSR–CLOB Bridge: maps Polymarket orderbook observables onto an LMSR
model so that every trade is priced through a rigorous microstructure lens.

Three responsibilities:
  1. Calibrate the LMSR liquidity parameter *b* from live orderbook depth.
  2. Compute execution-cost curves (slippage, impact) for arbitrary order sizes.
  3. Generate LMSR-based inefficiency signals that fuse with ML/LLM edges.

The CLOB is not literally an LMSR, but the mapping is well-defined:
  b_eff ≈ p(1-p) / (∂p/∂q)  where ∂p/∂q is estimated from the book.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .lmsr_engine import LMSREngine, MarketState, TradeQuote, InefficiencySignal
from .config import (
    LMSR_FEE_RATE,
    LMSR_MIN_EDGE,
    LMSR_MIN_EDGE_AFTER_FEES,
    LMSR_DEFAULT_B,
)

if TYPE_CHECKING:
    from .fetcher import PolymarketFetcher

log = logging.getLogger(__name__)

# Standard trade increments to quote (in shares)
QUOTE_DELTAS = [1, 5, 10, 25, 50, 100]


class LMSRAdapter:
    """
    Bridge between Polymarket's CLOB and the LMSR pricing model.

    For every market the pipeline considers, this adapter:
      - Calibrates an effective *b* from the live orderbook.
      - Quotes execution costs at standard size increments.
      - Produces InefficiencySignal objects comparable to ML/LLM edges.
    """

    def __init__(
        self,
        fee_rate: float = LMSR_FEE_RATE,
        min_edge: float = LMSR_MIN_EDGE,
        min_edge_after_fees: float = LMSR_MIN_EDGE_AFTER_FEES,
        default_b: float = LMSR_DEFAULT_B,
    ):
        self.engine = LMSREngine(
            b=default_b,
            fee_rate=fee_rate,
            min_edge=min_edge,
            min_edge_after_fees=min_edge_after_fees,
        )
        self.default_b = default_b

    # ------------------------------------------------------------------
    # b-Calibration from orderbook observables
    # ------------------------------------------------------------------

    def calibrate_b(
        self,
        mid_price: float,
        spread: float,
        liquidity_usd: float,
        volume_24h: float,
    ) -> float:
        """
        Estimate the effective LMSR liquidity parameter from CLOB data.

        Three independent estimators, combined conservatively:

        1. Spread-based:   b_spread  = p(1-p) / spread
           (From ∂p/∂q = p(1-p)/b for LMSR; spread ≈ ∂p for 1 unit.)

        2. Liquidity-based: b_liq = liquidity / (2·ln 2)
           (For binary: L_max = b·ln 2, and total book depth ≈ 2·L_max.)

        3. Volume-based:    b_vol = volume_24h / (24 * ln 2)
           (Hourly turnover as a softer proxy for available depth.)

        We take the *minimum* of the three — conservative means we
        *overestimate* our impact, which protects capital.
        """
        p = np.clip(mid_price, 0.02, 0.98)
        ln2 = math.log(2)

        b_spread = (p * (1 - p)) / max(spread, 1e-4)
        b_liquidity = max(liquidity_usd, 1.0) / (2 * ln2)
        b_volume = max(volume_24h, 1.0) / (24 * ln2)

        b_est = min(b_spread, b_liquidity, b_volume)
        b_est = np.clip(b_est, 5.0, 50_000.0)

        return float(b_est)

    def _build_state(self, mid_price: float, b: float) -> MarketState:
        """
        Construct a MarketState that reproduces the observed mid price.

        Given p_yes = mid_price, solve for q_yes - q_no:
          p_yes = e^(q_y/b) / (e^(q_y/b) + e^(q_n/b))
        Setting q_no = 0:
          q_yes = b · ln(p / (1-p))   (the log-odds, scaled by b)
        """
        p = np.clip(mid_price, 0.01, 0.99)
        q_yes = b * math.log(p / (1 - p))
        q_no = 0.0
        return MarketState.binary(q_yes=q_yes, q_no=q_no, b=b)

    # ------------------------------------------------------------------
    # Execution-cost curve
    # ------------------------------------------------------------------

    def quote_execution(
        self,
        mid_price: float,
        b: float,
        deltas: Optional[List[float]] = None,
        outcome: int = 0,
    ) -> List[TradeQuote]:
        """
        Quote a series of hypothetical trades at increasing sizes.
        Returns a cost/slippage curve useful for Kelly sizing.
        """
        state = self._build_state(mid_price, b)
        deltas = deltas or QUOTE_DELTAS
        return [self.engine.quote_trade(state, outcome, d) for d in deltas]

    def impact_cost(
        self,
        mid_price: float,
        b: float,
        order_size_usd: float,
    ) -> Dict[str, float]:
        """
        For a given USD order size, compute:
          - shares:           order_size / mid_price
          - lmsr_cost:        total cost through the LMSR curve
          - avg_fill_price:   lmsr_cost / shares
          - slippage_bps:     (avg_fill - mid) / mid * 10 000
          - edge_consumed_pct: fraction of edge eaten by impact

        This is used by the impact-adjusted Kelly sizer.
        """
        p = np.clip(mid_price, 0.01, 0.99)
        shares = order_size_usd / p
        if shares < 0.01:
            return {
                "shares": 0.0,
                "lmsr_cost": 0.0,
                "avg_fill_price": p,
                "slippage_bps": 0.0,
                "edge_consumed_pct": 0.0,
            }

        state = self._build_state(p, b)
        quote = self.engine.quote_trade(state, outcome=0, delta=shares)
        avg_fill = quote.cost / shares if shares > 0 else p

        slippage_bps = (avg_fill - p) / p * 10_000
        return {
            "shares": float(shares),
            "lmsr_cost": float(quote.cost),
            "avg_fill_price": float(avg_fill),
            "slippage_bps": float(slippage_bps),
            "edge_consumed_pct": float(abs(avg_fill - p) / max(abs(1 - p), 1e-6)),
        }

    # ------------------------------------------------------------------
    # Inefficiency signals (LMSR-based)
    # ------------------------------------------------------------------

    def detect_inefficiency(
        self,
        market_id: str,
        p_market: float,
        p_model: float,
        confidence: float,
        b: float,
    ) -> List[InefficiencySignal]:
        """
        Run the LMSR inefficiency detector using calibrated *b*.
        Accepts scalar prices (binary YES/NO) and returns signals.
        """
        engine = LMSREngine(
            b=b,
            fee_rate=self.engine.fee_rate,
            min_edge=self.engine.min_edge,
            min_edge_after_fees=self.engine.min_edge_after_fees,
        )
        market_prices = np.array([p_market, 1 - p_market])
        estimated_probs = np.array([p_model, 1 - p_model])
        confidences = np.array([confidence, confidence])

        return engine.detect_inefficiency(
            market_id=market_id,
            market_prices=market_prices,
            estimated_probs=estimated_probs,
            confidences=confidences,
        )

    # ------------------------------------------------------------------
    # Full market analysis (called from detector.py)
    # ------------------------------------------------------------------

    def analyze_market(
        self,
        market: dict,
        fetcher: "PolymarketFetcher",
        p_model: float,
        model_confidence: float,
    ) -> Dict:
        """
        End-to-end LMSR analysis for a single market.
        Returns a dict with calibrated b, cost curves, signals, and
        a scalar lmsr_edge suitable for fusion with ML/LLM edges.
        """
        token_id = (market.get("clobTokenIds") or [""])[0]
        if not token_id:
            return self._empty_analysis()

        prices = fetcher.get_best_prices(token_id)
        mid = prices["mid"]
        spread = prices["spread"]
        liquidity = float(market.get("liquidityClob", 0) or 0)
        volume = float(market.get("volume24hr", 0) or 0)

        b = self.calibrate_b(mid, spread, liquidity, volume)

        cost_curve = self.quote_execution(mid, b)

        signals = self.detect_inefficiency(
            market_id=market.get("id", ""),
            p_market=mid,
            p_model=p_model,
            confidence=model_confidence,
            b=b,
        )

        lmsr_edge = 0.0
        lmsr_side = "HOLD"
        lmsr_strength = "none"
        lmsr_ev = 0.0
        if signals:
            best = signals[0]
            lmsr_edge = abs(best.edge_after_fees)
            lmsr_side = best.side
            lmsr_strength = best.signal_strength
            lmsr_ev = best.expected_value

        ref_size = min(50.0, liquidity * 0.01) if liquidity > 0 else 10.0
        impact = self.impact_cost(mid, b, ref_size)

        max_maker_loss = self.engine.max_loss(2, b)
        capacity_usd = max_maker_loss * 0.5

        return {
            "b_calibrated": b,
            "mid_price": mid,
            "spread": spread,
            "lmsr_edge": lmsr_edge,
            "lmsr_side": lmsr_side,
            "lmsr_strength": lmsr_strength,
            "lmsr_ev": lmsr_ev,
            "impact_at_ref_size": impact,
            "cost_curve": [
                {
                    "delta": q.delta,
                    "cost": q.cost,
                    "slippage": q.slippage,
                    "price_after": q.price_after,
                }
                for q in cost_curve
            ],
            "capacity_usd": capacity_usd,
            "max_maker_loss": max_maker_loss,
        }

    @staticmethod
    def _empty_analysis() -> Dict:
        return {
            "b_calibrated": 0.0,
            "mid_price": 0.0,
            "spread": 1.0,
            "lmsr_edge": 0.0,
            "lmsr_side": "HOLD",
            "lmsr_strength": "none",
            "lmsr_ev": 0.0,
            "impact_at_ref_size": {},
            "cost_curve": [],
            "capacity_usd": 0.0,
            "max_maker_loss": 0.0,
        }
