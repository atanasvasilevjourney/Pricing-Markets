"""
Walk-forward backtest on resolved Polymarket markets.
Uses warproxxx/poly_data historical dataset.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Backtester:

    def __init__(self, trades_csv: str, markets_csv: str):
        """
        trades_csv:  Path to processed/trades.csv from poly_data repo
        markets_csv: Path to markets.csv from poly_data repo
        """
        self.trades = pd.read_csv(trades_csv, parse_dates=["timestamp"])
        self.markets = pd.read_csv(
            markets_csv, parse_dates=["createdAt", "closedTime"]
        )
        log.info(
            "Loaded %d trades across %d markets",
            len(self.trades), len(self.markets),
        )

    def backtest_henry_strategy(
        self,
        entry_min_price: float = 0.02,
        entry_max_price: float = 0.20,
        target_price: float = 0.30,
        stop_loss: float = 0.008,
        min_volume: float = 5_000,
    ) -> dict:
        """
        Simulate Henry's strategy on historical data.
        Entry: buy when YES price in [entry_min, entry_max].
        Exit:  sell when price reaches target OR stop_loss.
        """
        results = []

        for _, market in self.markets.iterrows():
            market_id = market.get("condition_id")
            market_trades = self.trades[
                self.trades["makerAssetId"] == market_id
            ].sort_values("timestamp")

            if market_trades.empty:
                continue

            for _, row in market_trades.iterrows():
                price = self._estimate_price(row)
                if not (entry_min_price <= price <= entry_max_price):
                    continue

                entry_price = price
                entry_time = row["timestamp"]
                outcome = None

                future_trades = market_trades[
                    market_trades["timestamp"] > entry_time
                ]
                for _, future in future_trades.iterrows():
                    future_price = self._estimate_price(future)
                    if future_price >= target_price:
                        outcome = {
                            "exit": "TAKE_PROFIT",
                            "exit_price": future_price,
                        }
                        break
                    if future_price <= stop_loss:
                        outcome = {
                            "exit": "STOP_LOSS",
                            "exit_price": future_price,
                        }
                        break

                if outcome is None:
                    resolution = market.get("answer1")
                    outcome = {
                        "exit": "RESOLUTION",
                        "exit_price": float(resolution) if resolution else 0,
                    }

                pnl = (outcome["exit_price"] - entry_price) / entry_price
                results.append({
                    "market_id": market_id,
                    "entry_price": entry_price,
                    "exit_price": outcome["exit_price"],
                    "exit_type": outcome["exit"],
                    "pnl_pct": pnl * 100,
                })
                break  # One position per market in backtest

        df = pd.DataFrame(results)
        if df.empty:
            return {"error": "No trades matched entry criteria"}

        wins = df[df["pnl_pct"] > 0]
        return {
            "n_trades": len(df),
            "win_rate": len(wins) / len(df) * 100,
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "median_pnl_pct": float(df["pnl_pct"].median()),
            "max_win": float(df["pnl_pct"].max()),
            "max_loss": float(df["pnl_pct"].min()),
            "total_return": float(df["pnl_pct"].sum()),
            "sharpe_approx": float(
                df["pnl_pct"].mean() / (df["pnl_pct"].std() + 1e-10)
            ),
            "exit_breakdown": df["exit_type"].value_counts().to_dict(),
        }

    @staticmethod
    def _estimate_price(trade_row) -> float:
        """Estimate price from a trade fill record."""
        maker_amt = float(trade_row.get("makerAmountFilled", 0) or 0)
        taker_amt = float(trade_row.get("takerAmountFilled", 0) or 0)
        if taker_amt > 0:
            return maker_amt / taker_amt
        return 0.5
