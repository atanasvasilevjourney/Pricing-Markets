"""
Production-grade Polymarket data fetcher.
Wraps all three APIs (CLOB, Gamma, Data) + Goldsky subgraph.
Implements rate limiting, retry with exponential backoff, and LRU caching.
"""

import time
import logging
from typing import Optional, List, Dict, Any

import requests

from .config import CLOB_API, GAMMA_API, DATA_API, GOLDSKY_URL

log = logging.getLogger(__name__)


class PolymarketFetcher:

    def __init__(self, rate_limit_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketBot/2.0",
        })
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0
        self._cache: Dict[str, tuple] = {}

    # ── Rate limiting ──────────────────────────────────────────────────
    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    # ── Retry wrapper ──────────────────────────────────────────────────
    def _get(
        self,
        url: str,
        params: Optional[dict] = None,
        retries: int = 3,
        cache_ttl: int = 60,
    ) -> Optional[Any]:
        cache_key = f"{url}:{params}"
        if cache_key in self._cache:
            cached_at, data = self._cache[cache_key]
            if time.time() - cached_at < cache_ttl:
                return data

        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                self._cache[cache_key] = (time.time(), data)
                return data
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                log.warning(
                    "[Retry %d/%d] %s: %s — waiting %ds",
                    attempt + 1, retries, url, e, wait,
                )
                time.sleep(wait)
        return None

    # ── Market discovery ───────────────────────────────────────────────
    def get_markets(
        self,
        limit: int = 100,
        order: str = "volume24hr",
        min_volume: float = 5_000,
        active_only: bool = True,
    ) -> List[dict]:
        params = {
            "limit": limit,
            "order": order,
            "ascending": "false",
            "active": "true" if active_only else "false",
        }
        raw = self._get(f"{GAMMA_API}/markets", params=params, cache_ttl=300)
        if not raw:
            return []

        markets = raw if isinstance(raw, list) else raw.get("markets", [])
        return [
            m for m in markets
            if float(m.get("volume24hr", 0) or 0) >= min_volume
        ]

    # ── Orderbook ──────────────────────────────────────────────────────
    def get_orderbook(self, token_id: str) -> Optional[dict]:
        return self._get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            cache_ttl=5,
        )

    def get_best_prices(self, token_id: str) -> Dict[str, float]:
        book = self.get_orderbook(token_id)
        if not book:
            return {"bid": 0, "ask": 1, "mid": 0.5, "spread": 1}
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        return {"bid": best_bid, "ask": best_ask, "mid": mid, "spread": spread}

    # ── Price history ──────────────────────────────────────────────────
    def get_price_history(
        self,
        token_id: str,
        interval: str = "1h",
        fidelity: int = 170,
    ) -> Optional[list]:
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        return self._get(
            f"{DATA_API}/prices-history", params=params, cache_ttl=300
        )

    # ── Trades (recent fills) ─────────────────────────────────────────
    def get_recent_trades(self, token_id: str, limit: int = 50) -> List[dict]:
        params = {"token_id": token_id, "limit": limit}
        raw = self._get(f"{CLOB_API}/trades", params=params, cache_ttl=30)
        return raw if isinstance(raw, list) else []

    # ── Multi-outcome market structure ─────────────────────────────────
    def get_event_markets(self, event_slug: str) -> List[dict]:
        raw = self._get(f"{GAMMA_API}/events", params={"slug": event_slug})
        if not raw:
            return []
        events = raw if isinstance(raw, list) else raw.get("events", [])
        return events[0].get("markets", []) if events else []

    # ── Goldsky on-chain fills ─────────────────────────────────────────
    def get_onchain_fills(
        self, maker_address: str, limit: int = 100
    ) -> List[dict]:
        query = """
        {
          orderFilledEvents(
            where: {maker: "%s"}
            orderBy: timestamp
            orderDirection: desc
            first: %d
          ) {
            timestamp
            maker
            makerAssetId
            makerAmountFilled
            taker
            takerAssetId
            takerAmountFilled
            transactionHash
          }
        }
        """ % (maker_address.lower(), limit)

        try:
            resp = requests.post(
                GOLDSKY_URL, json={"query": query}, timeout=30
            )
            data = resp.json()
            return data.get("data", {}).get("orderFilledEvents", [])
        except Exception as e:
            log.error("Goldsky error: %s", e)
            return []
