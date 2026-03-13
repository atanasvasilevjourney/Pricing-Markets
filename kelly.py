"""
Fractional Kelly Criterion position sizing.

Full Kelly:     f* = (p_true - p_market) / (1 - p_market)
Quarter Kelly:  f  = f* × 0.25

Quarter Kelly gives ~4% chance of halving bankroll before doubling,
compared to 33% for full Kelly.
"""

from .config import KELLY_FRACTION, MAX_SINGLE_POSITION_PCT, MAX_TOTAL_EXPOSURE_PCT


class KellyCriterion:

    @staticmethod
    def full_kelly(p_true: float, p_market: float) -> float:
        """
        Full Kelly fraction — never use directly, use position_size().
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
    def multi_position_check(
        positions: list,
        bankroll: float,
        max_total_pct: float = MAX_TOTAL_EXPOSURE_PCT,
    ) -> bool:
        """Ensure total open positions don't exceed max_total_pct of bankroll."""
        total = sum(p.get("size", 0) for p in positions)
        return (total / bankroll) <= max_total_pct if bankroll > 0 else False
