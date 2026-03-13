"""
Production CLOB execution via official Polymarket py-clob-client.
Requires: POLYGON_WALLET_PRIVATE_KEY env var.
"""

import logging
from typing import Optional

from .config import POLYGON_WALLET_PRIVATE_KEY, CHAIN_ID, CLOB_API
from .signals import Signal, TradeDecision

log = logging.getLogger(__name__)


class ExecutionEngine:

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        if not POLYGON_WALLET_PRIVATE_KEY:
            log.warning("No wallet key — execution engine in read-only mode")
            return
        try:
            from py_clob_client.client import ClobClient
            self.client = ClobClient(
                host=CLOB_API,
                key=POLYGON_WALLET_PRIVATE_KEY,
                chain_id=CHAIN_ID,
            )
            self.client.set_api_creds(
                self.client.create_or_derive_api_creds()
            )
            log.info("CLOB client initialised")
        except ImportError:
            log.warning("py-clob-client not installed — execution disabled")
        except Exception as e:
            log.error("CLOB client init failed: %s", e)

    def execute_trade(
        self, decision: TradeDecision, token_id: str
    ) -> dict:
        if decision.signal == Signal.HOLD:
            return {"status": "SKIPPED", "reason": "HOLD signal"}

        if not self.client:
            return {"status": "FAILED", "error": "No CLOB client available"}

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        limit_price = round(decision.current_price + 0.001, 4)
        shares = int(decision.position_size / decision.current_price)
        if shares < 1:
            return {"status": "SKIPPED", "reason": "Position too small"}

        order_args = OrderArgs(
            token_id=token_id,
            price=limit_price,
            size=shares,
            side=BUY,
        )

        try:
            resp = self.client.post_order(
                self.client.create_limit_order(order_args),
                OrderType.GTC,
            )
            return {
                "status": "PLACED",
                "order_id": resp.get("orderID"),
                "token_id": token_id,
                "direction": decision.direction,
                "shares": shares,
                "price": limit_price,
                "total_cost": shares * limit_price,
                "market_id": decision.market_id,
            }
        except Exception as e:
            log.error("Order failed: %s", e)
            return {"status": "FAILED", "error": str(e)}

    def cancel_order(self, order_id: str) -> Optional[dict]:
        if not self.client:
            return None
        try:
            return self.client.cancel(order_id)
        except Exception as e:
            log.error("Cancel failed for %s: %s", order_id, e)
            return None
