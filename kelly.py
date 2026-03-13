"""
Fractional Kelly Criterion — impact-aware position sizing.

Classic Kelly:   f* = (p_true - p_market) / (1 - p_market)
Quarter Kelly:   f  = f* × 0.25

This module adds an *impact-adjusted* variant that iteratively solves
for the largest position whose post-impact edge still justifies the
allocation.  Without this, naive Kelly over-sizes in thin markets and
the slippage eats the edge.

Math
----
Let s(q) = average fill price when buying q shares through the LMSR
cost curve.  The post-impact edge is:

    edge_net(q) = p_true - s(q)

We want the largest q such that:
    q / bankroll ≤ kelly_fraction · edge_net(q) / (1 - s(q))

Binary-searched to 0.1% precision.
"""

import logging
from typing import Callable, Optional

from .config import (
    KELLY_FRACTION,
    MAX_SINGLE_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    LMSR_MAX_IMPACT_PCT,
)

log = logging.getLogger(__name__)


class KellyCriterion:

    @staticmethod
    def full_kelly(p_true: float, p_market: float) -> float:
        """
        Full Kelly fraction for a binary bet.
        f* = (p_true - p_market) / (1 - p_market)
        """
        if p_true <= p_market or p_market >= 1.0:
            return 0.0
        return (p_true - p_market) / (1 - p_market)

    @staticmethod
    def position_size(
        p_true: float,
        p_market: float,
        confidence: float,
        bankroll: float,
        kelly_fraction: float = KELLY_FRACTION,
        max_pct: float = MAX_SINGLE_POSITION_PCT,
    ) -> float:
        """Quarter Kelly with confidence weighting. Hard cap at max_pct."""
        fk = KellyCriterion.full_kelly(p_true, p_market)
        size = fk * kelly_fraction * confidence * bankroll
        size = max(0.0, size)
        size = min(size, bankroll * max_pct)
        return round(size, 2)

    @staticmethod
    def position_size_impact_adjusted(
        p_true: float,
        p_market: float,
        confidence: float,
        bankroll: float,
        impact_cost_fn: Callable[[float], dict],
        kelly_fraction: float = KELLY_FRACTION,
        max_pct: float = MAX_SINGLE_POSITION_PCT,
        max_impact_pct: float = LMSR_MAX_IMPACT_PCT,
        n_iterations: int = 12,
    ) -> float:
        """
        Impact-adjusted Kelly sizing via binary search.

        impact_cost_fn(order_size_usd) -> dict with 'avg_fill_price'

        The loop finds the largest USD size where:
          1. Post-impact edge is still positive.
          2. Impact consumes < max_impact_pct of raw edge.
          3. Size ≤ kelly_fraction * f*(post-impact) * confidence * bankroll.
        """
        naive_size = KellyCriterion.position_size(
            p_true, p_market, confidence, bankroll,
            kelly_fraction, max_pct,
        )
        if naive_size <= 0:
            return 0.0

        raw_edge = p_true - p_market
        if raw_edge <= 0:
            return 0.0

        lo, hi = 0.0, naive_size
        best = 0.0

        for _ in range(n_iterations):
            mid = (lo + hi) / 2
            if mid < 0.01:
                break

            impact = impact_cost_fn(mid)
            avg_fill = impact.get("avg_fill_price", p_market)

            impact_cost = avg_fill - p_market
            if impact_cost / raw_edge > max_impact_pct:
                hi = mid
                continue

            net_edge = p_true - avg_fill
            if net_edge <= 0:
                hi = mid
                continue

            fk_net = net_edge / (1 - avg_fill) if avg_fill < 1.0 else 0.0
            kelly_size = fk_net * kelly_fraction * confidence * bankroll
            kelly_size = min(kelly_size, bankroll * max_pct)

            if mid <= kelly_size:
                best = mid
                lo = mid
            else:
                hi = mid

        return round(max(best, 0.0), 2)

    @staticmethod
    def multi_position_check(
        positions: list,
        bankroll: float,
        max_total_pct: float = MAX_TOTAL_EXPOSURE_PCT,
    ) -> bool:
        """Ensure total open positions don't exceed max_total_pct of bankroll."""
        total = sum(p.get("size", 0) for p in positions)
        return (total / bankroll) <= max_total_pct if bankroll > 0 else False
