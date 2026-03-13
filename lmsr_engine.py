"""
LMSR Pricing Engine -- Logarithmic Market Scoring Rule
======================================================

Vendored from: https://github.com/atanasvasilevjourney/lmsr-pricing-engine

Core equations:
  Cost function:  C(q) = b * ln(sum e^(qi/b))          -- Eq. (1)
  Max maker loss: L_max = b * ln(n)                     -- Eq. (2)
  Price (softmax): p_i(q) = e^(qi/b) / sum e^(qj/b)   -- Eq. (3)
  Trade cost:     cost = C(q_after) - C(q_before)       -- Eq. (4)

All computations use log-sum-exp for numerical stability.

References:
  Hanson, R. (2003). "Combinatorial Information Market Design."
  Hanson, R. (2007). "Logarithmic Market Scoring Rules for
    Modular Combinatorial Information Aggregation."
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarketState:
    """Current state of an LMSR market."""
    quantities: np.ndarray
    b: float
    n_outcomes: int

    @classmethod
    def binary(cls, q_yes: float, q_no: float, b: float) -> "MarketState":
        return cls(quantities=np.array([q_yes, q_no]), b=b, n_outcomes=2)


@dataclass
class TradeQuote:
    """Quote for a proposed trade."""
    outcome_index: int
    delta: float
    cost: float
    price_before: float
    price_after: float
    slippage: float
    market_state: MarketState


@dataclass
class InefficiencySignal:
    """
    Detected pricing inefficiency between model estimate and market price.
    Entry condition: |p_hat - p_market| > threshold (after fees).
    """
    market_id: str
    outcome_index: int
    market_price: float
    estimated_prob: float
    edge: float
    edge_after_fees: float
    expected_value: float
    confidence: float
    signal_strength: str
    side: str


class LMSREngine:
    """
    Core LMSR pricing engine with log-sum-exp numerical stability.

    Parameters
    ----------
    b : float
        Liquidity parameter. Higher b = more liquidity, less price
        impact per trade, but higher max loss for the market maker.
    fee_rate : float
        Trading fee as a fraction (e.g., 0.02 = 2%).
    min_edge : float
        Minimum |p_hat - p_market| to generate a signal.
    min_edge_after_fees : float
        Minimum edge after deducting fees.
    """

    def __init__(
        self,
        b: float = 100.0,
        fee_rate: float = 0.02,
        min_edge: float = 0.03,
        min_edge_after_fees: float = 0.01,
    ):
        self.b = b
        self.fee_rate = fee_rate
        self.min_edge = min_edge
        self.min_edge_after_fees = min_edge_after_fees

    # -- Eq. (1): Cost Function C(q) = b * ln(sum e^(qi/b)) -----------

    def cost(self, quantities: np.ndarray, b: Optional[float] = None) -> float:
        b = b or self.b
        scaled = quantities / b
        max_scaled = np.max(scaled)
        return b * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))

    # -- Eq. (2): Max Market Maker Loss --------------------------------

    def max_loss(self, n_outcomes: int, b: Optional[float] = None) -> float:
        b = b or self.b
        return b * math.log(n_outcomes)

    # -- Eq. (3): Price Function (Softmax) -----------------------------

    def prices(self, quantities: np.ndarray, b: Optional[float] = None) -> np.ndarray:
        b = b or self.b
        scaled = quantities / b
        max_scaled = np.max(scaled)
        exp_shifted = np.exp(scaled - max_scaled)
        return exp_shifted / np.sum(exp_shifted)

    def price(self, quantities: np.ndarray, outcome: int,
              b: Optional[float] = None) -> float:
        return float(self.prices(quantities, b)[outcome])

    # -- Eq. (4): Trade Cost = C(q_after) - C(q_before) ---------------

    def trade_cost(self, quantities: np.ndarray, outcome: int,
                   delta: float, b: Optional[float] = None) -> float:
        b = b or self.b
        cost_before = self.cost(quantities, b)
        new_quantities = quantities.copy()
        new_quantities[outcome] += delta
        cost_after = self.cost(new_quantities, b)
        return cost_after - cost_before

    def quote_trade(self, state: MarketState, outcome: int,
                    delta: float) -> TradeQuote:
        price_before = self.price(state.quantities, outcome, state.b)
        cost = self.trade_cost(state.quantities, outcome, delta, state.b)

        new_quantities = state.quantities.copy()
        new_quantities[outcome] += delta
        new_state = MarketState(
            quantities=new_quantities,
            b=state.b,
            n_outcomes=state.n_outcomes,
        )
        price_after = self.price(new_quantities, outcome, state.b)

        return TradeQuote(
            outcome_index=outcome,
            delta=delta,
            cost=cost,
            price_before=price_before,
            price_after=price_after,
            slippage=abs(price_after - price_before),
            market_state=new_state,
        )

    # -- Inefficiency Detection ----------------------------------------

    def detect_inefficiency(
        self,
        market_id: str,
        market_prices: np.ndarray,
        estimated_probs: np.ndarray,
        confidences: np.ndarray,
    ) -> List[InefficiencySignal]:
        signals: List[InefficiencySignal] = []

        for i in range(len(market_prices)):
            p_market = market_prices[i]
            p_hat = estimated_probs[i]
            confidence = confidences[i]
            edge = p_hat - p_market

            if edge > 0:
                side = "BUY"
                ev = p_hat * (1.0 - p_market) - (1.0 - p_hat) * p_market
                edge_after_fees = edge - self.fee_rate
            else:
                side = "SELL"
                ev = (1.0 - p_hat) * p_market - p_hat * (1.0 - p_market)
                edge_after_fees = abs(edge) - self.fee_rate

            if abs(edge) < self.min_edge:
                continue
            if edge_after_fees < self.min_edge_after_fees:
                continue

            abs_edge = abs(edge)
            if abs_edge > 0.10:
                strength = "strong"
            elif abs_edge > 0.05:
                strength = "moderate"
            else:
                strength = "weak"

            signals.append(InefficiencySignal(
                market_id=market_id,
                outcome_index=i,
                market_price=p_market,
                estimated_prob=p_hat,
                edge=edge,
                edge_after_fees=edge_after_fees,
                expected_value=ev,
                confidence=confidence,
                signal_strength=strength,
                side=side,
            ))

        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def validate_prices(self, prices: np.ndarray, tol: float = 1e-6) -> bool:
        return (
            abs(np.sum(prices) - 1.0) < tol
            and np.all(prices > 0)
            and np.all(prices < 1)
        )
