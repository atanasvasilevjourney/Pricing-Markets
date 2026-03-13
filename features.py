"""
Feature extraction — exactly 10 signals in fixed order.
Dimension consistency is critical: train and predict must always see 10 features.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fetcher import PolymarketFetcher


class FeatureExtractor:

    FEATURE_NAMES = [
        "current_price",    # 0: Current YES mid price (0–1)
        "volume_24h",       # 1: 24h trading volume (USD, log-transformed)
        "liquidity",        # 2: Available liquidity (USD, log-transformed)
        "rsi",              # 3: Relative Strength Index (0–1)
        "momentum",         # 4: Short-term price momentum
        "order_imbalance",  # 5: (buy_vol - sell_vol) / total_vol
        "volatility",       # 6: Price standard deviation (recent)
        "change_1d",        # 7: 24h price change (signed)
        "change_7d",        # 8: 7d price change (signed)
        "spread",           # 9: Best ask - best bid
    ]
    N_FEATURES = 10

    def extract(self, market: dict, fetcher: "PolymarketFetcher") -> np.ndarray:
        """
        Extract feature vector for a market.
        Returns ndarray of shape (10,) — ALWAYS 10 features.
        """
        token_id = (market.get("clobTokenIds") or [""])[0]
        if not token_id:
            return self._zero_vector()

        prices = fetcher.get_best_prices(token_id)
        current_price = prices["mid"]
        spread = prices["spread"]

        volume_24h = float(market.get("volume24hr", 0) or 0)
        liquidity = float(market.get("liquidityClob", 0) or 0)

        history = fetcher.get_price_history(
            token_id, interval="1h", fidelity=170
        )
        price_series = self._extract_price_series(history)

        rsi = self._calculate_rsi(price_series, period=14)
        momentum = self._calculate_momentum(price_series, window=6)
        volatility = self._calculate_volatility(price_series, window=24)
        change_1d = self._calculate_change(price_series, periods=24)
        change_7d = self._calculate_change(price_series, periods=168)

        trades = fetcher.get_recent_trades(token_id, limit=50)
        order_imbalance = self._calculate_order_imbalance(trades)

        features = np.array(
            [
                current_price,
                np.log1p(volume_24h),
                np.log1p(liquidity),
                rsi,
                momentum,
                order_imbalance,
                volatility,
                change_1d,
                change_7d,
                spread,
            ],
            dtype=np.float32,
        )

        assert features.shape == (self.N_FEATURES,), (
            f"Feature shape mismatch: {features.shape} != ({self.N_FEATURES},)"
        )

        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    # ── Signal calculations ────────────────────────────────────────────

    @staticmethod
    def _extract_price_series(history) -> np.ndarray:
        if not history:
            return np.array([0.5], dtype=np.float32)
        if isinstance(history, list):
            prices = [
                float(p.get("p", p.get("price", 0.5))) for p in history
            ]
        elif isinstance(history, dict):
            prices = [float(p) for p in history.get("prices", [0.5])]
        else:
            return np.array([0.5], dtype=np.float32)
        return np.array(prices, dtype=np.float32)

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI normalised to [0, 1]. 0.3 = oversold, 0.7 = overbought."""
        if len(prices) < period + 1:
            return 0.5
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) + 1e-10
        avg_loss = np.mean(losses[-period:]) + 1e-10
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return float(np.clip(rsi, 0, 1))

    @staticmethod
    def _calculate_momentum(prices: np.ndarray, window: int = 6) -> float:
        if len(prices) < window + 1:
            return 0.0
        return float(prices[-1] - prices[-window - 1])

    @staticmethod
    def _calculate_volatility(prices: np.ndarray, window: int = 24) -> float:
        if len(prices) < 2:
            return 0.0
        return float(np.std(prices[-window:]))

    @staticmethod
    def _calculate_change(prices: np.ndarray, periods: int) -> float:
        if len(prices) < periods + 1:
            return 0.0
        return float(prices[-1] - prices[-(periods + 1)])

    @staticmethod
    def _calculate_order_imbalance(trades: list) -> float:
        """
        (buy_volume - sell_volume) / total_volume.
        Positive → net buying pressure → price likely to rise.
        """
        if not trades:
            return 0.0
        buy_vol = sell_vol = 0.0
        for t in trades:
            maker_asset = str(t.get("makerAssetId", "")).upper()
            amount = float(t.get("makerAmountFilled", 0) or 0)
            if "USDC" in maker_asset:
                sell_vol += amount
            else:
                buy_vol += amount
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

    def _zero_vector(self) -> np.ndarray:
        return np.zeros(self.N_FEATURES, dtype=np.float32)
