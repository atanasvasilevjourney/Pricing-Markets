"""
Combinatorial arbitrage scanner.
Detects both intra-market and inter-market mathematical mispricing.
Source: arXiv:2508.03474 — $40M documented profit, Apr 2024–Apr 2025.
"""

import logging
from typing import Optional

from .fetcher import PolymarketFetcher

log = logging.getLogger(__name__)

WINNER_FEE = 0.02  # Polymarket 2% winner fee


class CombinatorialArbScanner:

    MAX_ARB_COST = 0.98  # YES + NO total must be ≤ $0.98

    def scan_intra_market(
        self, market: dict, fetcher: PolymarketFetcher
    ) -> Optional[dict]:
        """
        Type A: YES + NO < 1.00 within a single binary market.
        Guaranteed $1.00 at resolution minus fees.
        """
        tokens = market.get("clobTokenIds", [])
        if len(tokens) < 2:
            return None

        yes_id, no_id = tokens[0], tokens[1]
        yes_prices = fetcher.get_best_prices(yes_id)
        no_prices = fetcher.get_best_prices(no_id)

        yes_ask = yes_prices["ask"]
        no_ask = no_prices["ask"]
        total = yes_ask + no_ask

        if total < self.MAX_ARB_COST:
            profit_pct = (1.0 - total - WINNER_FEE) / total * 100
            if profit_pct > 0:
                return {
                    "type": "INTRA_MARKET_ARB",
                    "market_id": market.get("id"),
                    "question": market.get("question"),
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    "total_cost": total,
                    "profit_pct": profit_pct,
                    "action": "BUY_BOTH",
                }
        return None

    def scan_multi_outcome(
        self, event_slug: str, fetcher: PolymarketFetcher
    ) -> Optional[dict]:
        """
        Type B: Multi-outcome markets (A / B / C / …).
        Sum of all outcome asks should = 1.00.
        """
        markets = fetcher.get_event_markets(event_slug)
        if len(markets) < 2:
            return None

        total_best_ask = 0.0
        outcomes = []

        for m in markets:
            tokens = m.get("clobTokenIds", [])
            if not tokens:
                continue
            prices = fetcher.get_best_prices(tokens[0])
            total_best_ask += prices["ask"]
            outcomes.append({
                "question": m.get("question"),
                "ask": prices["ask"],
                "token_id": tokens[0],
            })

        if total_best_ask < self.MAX_ARB_COST:
            profit_pct = (1.0 - total_best_ask - WINNER_FEE) / total_best_ask * 100
            if profit_pct > 0:
                return {
                    "type": "COMBINATORIAL_ARB",
                    "event_slug": event_slug,
                    "outcomes": outcomes,
                    "total_cost": total_best_ask,
                    "profit_pct": profit_pct,
                    "action": "BUY_ALL_OUTCOMES",
                }
        return None
