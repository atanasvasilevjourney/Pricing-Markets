"""
Walk-forward backtest on resolved Polymarket markets.

Two modes:
  1. Naive:      assumes fills at mid price (original behaviour).
  2. LMSR-aware: models slippage through the LMSR cost curve,
                 applies impact-adjusted Kelly sizing, and computes
                 realistic PnL net of fees and execution cost.

Uses warproxxx/poly_data historical dataset.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from .lmsr_engine import LMSREngine, MarketState

log = logging.getLogger(__name__)

WINNER_FEE = 0.02


class Backtester:

    def __init__(self, trades_csv: str, markets_csv: str):
        self.trades = pd.read_csv(trades_csv, parse_dates=["timestamp"])
        self.markets = pd.read_csv(
            markets_csv, parse_dates=["createdAt", "closedTime"]
        )
        log.info(
            "Loaded %d trades across %d markets",
            len(self.trades), len(self.markets),
        )

    # ------------------------------------------------------------------
    # Mode 1: Original naive Henry backtest
    # ------------------------------------------------------------------

    def backtest_henry_strategy(
        self,
        entry_min_price: float = 0.02,
        entry_max_price: float = 0.20,
        target_price: float = 0.30,
        stop_loss: float = 0.008,
        min_volume: float = 5_000,
    ) -> dict:
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
                break

        return self._summarize(results, label="henry_naive")

    # ------------------------------------------------------------------
    # Mode 2: LMSR-aware backtest with impact modelling
    # ------------------------------------------------------------------

    def backtest_lmsr_aware(
        self,
        entry_min_price: float = 0.02,
        entry_max_price: float = 0.20,
        target_price: float = 0.30,
        stop_loss: float = 0.008,
        bankroll: float = 500.0,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
    ) -> dict:
        """
        LMSR-aware backtest:
          - For each candidate trade, calibrate b from local liquidity.
          - Model entry and exit through the LMSR cost curve.
          - Size with impact-adjusted quarter-Kelly.
          - Track running bankroll and cumulative PnL.
        """
        results = []
        running_bankroll = bankroll

        for _, market in self.markets.iterrows():
            market_id = market.get("condition_id")
            market_trades = self.trades[
                self.trades["makerAssetId"] == market_id
            ].sort_values("timestamp")

            if market_trades.empty:
                continue

            liquidity = float(market.get("volume", 0) or 0)

            for _, row in market_trades.iterrows():
                price = self._estimate_price(row)
                if not (entry_min_price <= price <= entry_max_price):
                    continue

                b = self._calibrate_b_backtest(price, liquidity)
                engine = LMSREngine(b=b, fee_rate=WINNER_FEE)

                position_usd = min(
                    running_bankroll * max_position_pct,
                    running_bankroll * kelly_fraction * 0.5,
                )
                if position_usd < 1.0:
                    continue

                shares = position_usd / price
                state = self._build_state(price, b)
                entry_quote = engine.quote_trade(state, outcome=0, delta=shares)
                effective_entry = entry_quote.cost / shares if shares > 0 else price

                entry_time = row["timestamp"]
                outcome = None

                future_trades = market_trades[
                    market_trades["timestamp"] > entry_time
                ]
                for _, future in future_trades.iterrows():
                    future_price = self._estimate_price(future)
                    if future_price >= target_price:
                        outcome = {"exit": "TAKE_PROFIT", "raw_exit": future_price}
                        break
                    if future_price <= stop_loss:
                        outcome = {"exit": "STOP_LOSS", "raw_exit": future_price}
                        break

                if outcome is None:
                    resolution = market.get("answer1")
                    outcome = {
                        "exit": "RESOLUTION",
                        "raw_exit": float(resolution) if resolution else 0,
                    }

                exit_state = self._build_state(outcome["raw_exit"], b)
                exit_quote = engine.quote_trade(
                    exit_state, outcome=0, delta=-shares,
                )
                effective_exit = -exit_quote.cost / shares if shares > 0 else outcome["raw_exit"]

                gross_pnl = (effective_exit - effective_entry) * shares
                fee = max(gross_pnl, 0) * WINNER_FEE
                net_pnl = gross_pnl - fee

                running_bankroll += net_pnl

                results.append({
                    "market_id": market_id,
                    "entry_price": price,
                    "effective_entry": effective_entry,
                    "exit_price": outcome["raw_exit"],
                    "effective_exit": effective_exit,
                    "exit_type": outcome["exit"],
                    "shares": shares,
                    "position_usd": position_usd,
                    "gross_pnl": gross_pnl,
                    "fee": fee,
                    "net_pnl": net_pnl,
                    "pnl_pct": (net_pnl / position_usd * 100) if position_usd > 0 else 0,
                    "b_calibrated": b,
                    "entry_slippage_bps": (effective_entry - price) / price * 10_000,
                    "bankroll_after": running_bankroll,
                })
                break

        return self._summarize_lmsr(results, bankroll, running_bankroll)

    # ------------------------------------------------------------------
    # Comparative runner
    # ------------------------------------------------------------------

    def compare_modes(self, **kwargs) -> dict:
        """Run both modes and return side-by-side metrics."""
        naive = self.backtest_henry_strategy(**kwargs)
        lmsr = self.backtest_lmsr_aware(**kwargs)
        return {
            "naive": naive,
            "lmsr_aware": lmsr,
            "improvement": {
                "sharpe_delta": (
                    lmsr.get("sharpe_approx", 0) - naive.get("sharpe_approx", 0)
                ),
                "winrate_delta": (
                    lmsr.get("win_rate", 0) - naive.get("win_rate", 0)
                ),
                "avg_pnl_delta": (
                    lmsr.get("avg_pnl_pct", 0) - naive.get("avg_pnl_pct", 0)
                ),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_price(trade_row) -> float:
        maker_amt = float(trade_row.get("makerAmountFilled", 0) or 0)
        taker_amt = float(trade_row.get("takerAmountFilled", 0) or 0)
        if taker_amt > 0:
            return maker_amt / taker_amt
        return 0.5

    @staticmethod
    def _calibrate_b_backtest(mid_price: float, liquidity: float) -> float:
        p = np.clip(mid_price, 0.02, 0.98)
        ln2 = math.log(2)
        b_liq = max(liquidity, 100.0) / (2 * ln2)
        b_spread = (p * (1 - p)) / 0.02
        return float(np.clip(min(b_liq, b_spread), 5.0, 50_000.0))

    @staticmethod
    def _build_state(mid_price: float, b: float) -> MarketState:
        p = np.clip(mid_price, 0.01, 0.99)
        q_yes = b * math.log(p / (1 - p))
        return MarketState.binary(q_yes=q_yes, q_no=0.0, b=b)

    @staticmethod
    def _summarize(results: list, label: str = "") -> dict:
        df = pd.DataFrame(results)
        if df.empty:
            return {"error": "No trades matched entry criteria", "mode": label}
        wins = df[df["pnl_pct"] > 0]
        return {
            "mode": label,
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
    def _summarize_lmsr(
        results: list,
        initial_bankroll: float,
        final_bankroll: float,
    ) -> dict:
        df = pd.DataFrame(results)
        if df.empty:
            return {"error": "No trades matched", "mode": "lmsr_aware"}

        wins = df[df["net_pnl"] > 0]
        total_fees = float(df["fee"].sum())
        total_slippage = float(df["entry_slippage_bps"].mean())

        return {
            "mode": "lmsr_aware",
            "n_trades": len(df),
            "win_rate": len(wins) / len(df) * 100,
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "median_pnl_pct": float(df["pnl_pct"].median()),
            "max_win": float(df["pnl_pct"].max()),
            "max_loss": float(df["pnl_pct"].min()),
            "total_return_pct": (final_bankroll - initial_bankroll) / initial_bankroll * 100,
            "final_bankroll": final_bankroll,
            "total_fees_paid": total_fees,
            "avg_entry_slippage_bps": total_slippage,
            "sharpe_approx": float(
                df["pnl_pct"].mean() / (df["pnl_pct"].std() + 1e-10)
            ),
            "exit_breakdown": df["exit_type"].value_counts().to_dict(),
            "max_drawdown_pct": float(
                _max_drawdown(df["bankroll_after"].values, initial_bankroll)
            ),
        }


def _max_drawdown(equity_curve: np.ndarray, initial: float) -> float:
    """Peak-to-trough max drawdown as a percentage."""
    curve = np.concatenate([[initial], equity_curve])
    peak = np.maximum.accumulate(curve)
    dd = (peak - curve) / peak * 100
    return float(np.max(dd)) if len(dd) > 0 else 0.0
