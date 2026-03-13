#!/usr/bin/env python3
"""
Backtest runner — compares naive Henry vs LMSR-aware strategies.

Usage:
    python scripts/run_backtests.py --trades data/trades.csv --markets data/markets.csv

If no CSV files exist yet, the script generates synthetic data so you can
validate the pipeline end-to-end before plugging in real poly_data.

Outputs:
    output/backtest_comparison.png   — side-by-side metrics
    output/equity_curves.png         — cumulative PnL over trades
    output/exit_breakdown.png        — pie charts of exit types
    output/slippage_histogram.png    — distribution of entry slippage
    output/sensitivity.png           — parameter sweep (entry zone)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))
from Polymarket.backtester import Backtester

OUTPUT_DIR = Path("output")

# ── Synthetic data generator ──────────────────────────────────────────

def generate_synthetic_data(
    n_markets: int = 80,
    trades_per_market: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Generate plausible synthetic trades + markets data so the backtest
    pipeline can be validated without real poly_data CSVs.
    """
    rng = np.random.default_rng(seed)
    all_trades = []
    all_markets = []

    for i in range(n_markets):
        cid = f"MKT-{i:04d}"
        start_price = rng.uniform(0.03, 0.90)
        resolved_yes = rng.random() < start_price

        prices = [start_price]
        for _ in range(trades_per_market - 1):
            drift = 0.002 if resolved_yes else -0.002
            shock = rng.normal(0, 0.015)
            p = np.clip(prices[-1] + drift + shock, 0.005, 0.995)
            prices.append(p)

        base_ts = pd.Timestamp("2024-06-01") + pd.Timedelta(hours=i * 6)
        for j, p in enumerate(prices):
            maker_amt = p * rng.uniform(50, 500)
            taker_amt = maker_amt / p if p > 0 else 1
            all_trades.append({
                "timestamp": base_ts + pd.Timedelta(minutes=j * 15),
                "makerAssetId": cid,
                "makerAmountFilled": round(maker_amt, 4),
                "takerAmountFilled": round(taker_amt, 4),
            })

        volume = rng.uniform(3_000, 200_000)
        all_markets.append({
            "condition_id": cid,
            "question": f"Synthetic market {i}",
            "createdAt": base_ts,
            "closedTime": base_ts + pd.Timedelta(days=30),
            "volume": volume,
            "answer1": 1.0 if resolved_yes else 0.0,
        })

    trades_df = pd.DataFrame(all_trades)
    markets_df = pd.DataFrame(all_markets)
    return trades_df, markets_df


# ── Plotting functions ────────────────────────────────────────────────

COLORS = {
    "naive": "#4A90D9",
    "lmsr": "#E5573F",
    "accent": "#2ECC71",
    "bg": "#F8F9FA",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}


def _style_ax(ax, title=""):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.4, color=COLORS["grid"], linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"], pad=10)


def plot_comparison(naive: dict, lmsr: dict, path: Path):
    metrics = ["win_rate", "avg_pnl_pct", "sharpe_approx"]
    labels = ["Win Rate (%)", "Avg PnL (%)", "Sharpe (approx)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Naive Henry  vs  LMSR-Aware", fontsize=16, fontweight="bold",
                 color=COLORS["text"], y=1.02)

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [naive.get(metric, 0), lmsr.get(metric, 0)]
        bars = ax.bar(["Naive", "LMSR-Aware"], vals,
                      color=[COLORS["naive"], COLORS["lmsr"]], width=0.5, edgecolor="white")
        _style_ax(ax, label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(abs(v), 1),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_equity_curves(naive_results: list, lmsr_results: list, path: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    _style_ax(ax, "Cumulative PnL (%) — Trade-by-Trade")

    if naive_results:
        naive_cum = np.cumsum([r["pnl_pct"] for r in naive_results])
        ax.plot(naive_cum, color=COLORS["naive"], linewidth=2, label="Naive Henry")

    if lmsr_results:
        lmsr_cum = np.cumsum([r["pnl_pct"] for r in lmsr_results])
        ax.plot(lmsr_cum, color=COLORS["lmsr"], linewidth=2, label="LMSR-Aware")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trade #", fontsize=11)
    ax.set_ylabel("Cumulative PnL (%)", fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_exit_breakdown(naive: dict, lmsr: dict, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, color_start in [
        (ax1, naive.get("exit_breakdown", {}), "Naive Henry", 0.3),
        (ax2, lmsr.get("exit_breakdown", {}), "LMSR-Aware", 0.6),
    ]:
        if data:
            labels = list(data.keys())
            sizes = list(data.values())
            cmap = plt.cm.Set2
            colors = [cmap(color_start + i * 0.15) for i in range(len(labels))]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.0f%%", colors=colors,
                textprops={"fontsize": 10}, startangle=90,
            )
            for t in autotexts:
                t.set_fontweight("bold")
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])

    fig.suptitle("Exit Type Breakdown", fontsize=15, fontweight="bold",
                 color=COLORS["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_slippage_histogram(lmsr_results: list, path: Path):
    if not lmsr_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    slippages = [r.get("entry_slippage_bps", 0) for r in lmsr_results]
    ax1.hist(slippages, bins=30, color=COLORS["lmsr"], edgecolor="white", alpha=0.85)
    _style_ax(ax1, "Entry Slippage Distribution (bps)")
    ax1.set_xlabel("Slippage (basis points)")
    ax1.set_ylabel("Frequency")
    ax1.axvline(np.mean(slippages), color=COLORS["text"], linestyle="--",
                label=f"Mean: {np.mean(slippages):.0f} bps")
    ax1.legend()

    b_vals = [r.get("b_calibrated", 0) for r in lmsr_results if r.get("b_calibrated", 0) > 0]
    if b_vals:
        ax2.hist(b_vals, bins=30, color=COLORS["accent"], edgecolor="white", alpha=0.85)
        _style_ax(ax2, "Calibrated b Distribution")
        ax2.set_xlabel("b (liquidity parameter)")
        ax2.set_ylabel("Frequency")
        ax2.axvline(np.median(b_vals), color=COLORS["text"], linestyle="--",
                    label=f"Median: {np.median(b_vals):.1f}")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sensitivity(bt: Backtester, path: Path):
    """Sweep entry_max_price and target_price, plot Sharpe surface."""
    entry_maxes = np.arange(0.10, 0.35, 0.05)
    targets = np.arange(0.20, 0.55, 0.05)
    sharpe_naive = np.zeros((len(entry_maxes), len(targets)))
    sharpe_lmsr = np.zeros((len(entry_maxes), len(targets)))

    for i, em in enumerate(entry_maxes):
        for j, tgt in enumerate(targets):
            if tgt <= em:
                sharpe_naive[i, j] = np.nan
                sharpe_lmsr[i, j] = np.nan
                continue
            try:
                rn = bt.backtest_henry_strategy(
                    entry_min_price=0.02, entry_max_price=em,
                    target_price=tgt, stop_loss=0.008,
                )
                sharpe_naive[i, j] = rn.get("sharpe_approx", 0)
            except Exception:
                sharpe_naive[i, j] = np.nan
            try:
                rl = bt.backtest_lmsr_aware(
                    entry_min_price=0.02, entry_max_price=em,
                    target_price=tgt, stop_loss=0.008, bankroll=500,
                )
                sharpe_lmsr[i, j] = rl.get("sharpe_approx", 0)
            except Exception:
                sharpe_lmsr[i, j] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for ax, data, title in [
        (ax1, sharpe_naive, "Naive Henry — Sharpe"),
        (ax2, sharpe_lmsr, "LMSR-Aware — Sharpe"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([f"{t:.2f}" for t in targets], rotation=45)
        ax.set_yticks(range(len(entry_maxes)))
        ax.set_yticklabels([f"{e:.2f}" for e in entry_maxes])
        ax.set_xlabel("Target Price")
        ax.set_ylabel("Entry Max Price")
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Parameter Sensitivity — Sharpe Ratio", fontsize=15,
                 fontweight="bold", color=COLORS["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest runner")
    parser.add_argument("--trades", type=str, default=None,
                        help="Path to trades.csv (poly_data format)")
    parser.add_argument("--markets", type=str, default=None,
                        help="Path to markets.csv (poly_data format)")
    parser.add_argument("--bankroll", type=float, default=500.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data even if CSVs exist")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip parameter sweep (slow)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load or generate data
    use_synthetic = args.synthetic or (args.trades is None and args.markets is None)

    if use_synthetic:
        print("Generating synthetic data (80 markets, 200 trades each)...")
        trades_df, markets_df = generate_synthetic_data()
        trades_path = OUTPUT_DIR / "synth_trades.csv"
        markets_path = OUTPUT_DIR / "synth_markets.csv"
        trades_df.to_csv(trades_path, index=False)
        markets_df.to_csv(markets_path, index=False)
        print(f"  Saved {trades_path} ({len(trades_df)} rows)")
        print(f"  Saved {markets_path} ({len(markets_df)} rows)")
    else:
        trades_path = args.trades
        markets_path = args.markets
        if not Path(trades_path).exists():
            print(f"ERROR: {trades_path} not found")
            sys.exit(1)
        if not Path(markets_path).exists():
            print(f"ERROR: {markets_path} not found")
            sys.exit(1)

    bt = Backtester(str(trades_path), str(markets_path))

    # Run backtests
    print("\n=== Running Naive Henry Strategy ===")
    naive = bt.backtest_henry_strategy()
    print(f"  Trades: {naive.get('n_trades', 0)}")
    print(f"  Win rate: {naive.get('win_rate', 0):.1f}%")
    print(f"  Avg PnL: {naive.get('avg_pnl_pct', 0):.2f}%")
    print(f"  Sharpe: {naive.get('sharpe_approx', 0):.3f}")

    print("\n=== Running LMSR-Aware Strategy ===")
    lmsr = bt.backtest_lmsr_aware(bankroll=args.bankroll)
    print(f"  Trades: {lmsr.get('n_trades', 0)}")
    print(f"  Win rate: {lmsr.get('win_rate', 0):.1f}%")
    print(f"  Avg PnL: {lmsr.get('avg_pnl_pct', 0):.2f}%")
    print(f"  Sharpe: {lmsr.get('sharpe_approx', 0):.3f}")
    print(f"  Max DD: {lmsr.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Final bankroll: ${lmsr.get('final_bankroll', 0):.2f}")
    print(f"  Avg slippage: {lmsr.get('avg_entry_slippage_bps', 0):.0f} bps")

    # Collect per-trade results for curves
    naive_results = _rerun_for_results(bt, "naive")
    lmsr_results = _rerun_for_results(bt, "lmsr", bankroll=args.bankroll)

    # Generate plots
    print("\n=== Generating Charts ===")
    plot_comparison(naive, lmsr, OUTPUT_DIR / "backtest_comparison.png")
    plot_equity_curves(naive_results, lmsr_results, OUTPUT_DIR / "equity_curves.png")
    plot_exit_breakdown(naive, lmsr, OUTPUT_DIR / "exit_breakdown.png")
    plot_slippage_histogram(lmsr_results, OUTPUT_DIR / "slippage_histogram.png")

    if not args.skip_sensitivity:
        print("\n=== Running Parameter Sensitivity Sweep ===")
        plot_sensitivity(bt, OUTPUT_DIR / "sensitivity.png")

    # Summary table
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Metric':<25} {'Naive':>12} {'LMSR-Aware':>12} {'Delta':>12}")
    print("-" * 63)
    for metric, label in [
        ("win_rate", "Win Rate (%)"),
        ("avg_pnl_pct", "Avg PnL (%)"),
        ("sharpe_approx", "Sharpe"),
        ("n_trades", "# Trades"),
    ]:
        nv = naive.get(metric, 0)
        lv = lmsr.get(metric, 0)
        d = lv - nv
        print(f"{label:<25} {nv:>12.2f} {lv:>12.2f} {d:>+12.2f}")

    print(f"\nAll charts saved to {OUTPUT_DIR.resolve()}/")


def _rerun_for_results(bt: Backtester, mode: str, bankroll: float = 500.0) -> list:
    """Re-run backtest to get per-trade result dicts for plotting."""
    import math
    from Polymarket.lmsr_engine import LMSREngine, MarketState

    results = []
    running_bankroll = bankroll

    for _, market in bt.markets.iterrows():
        market_id = market.get("condition_id")
        market_trades = bt.trades[
            bt.trades["makerAssetId"] == market_id
        ].sort_values("timestamp")
        if market_trades.empty:
            continue

        for _, row in market_trades.iterrows():
            price = bt._estimate_price(row)
            if not (0.02 <= price <= 0.20):
                continue

            if mode == "lmsr":
                liquidity = float(market.get("volume", 0) or 0)
                b = bt._calibrate_b_backtest(price, liquidity)
                position_usd = min(running_bankroll * 0.05, running_bankroll * 0.25 * 0.5)
                if position_usd < 1:
                    continue
                shares = position_usd / price
                engine = LMSREngine(b=b, fee_rate=0.02)
                state = bt._build_state(price, b)
                entry_quote = engine.quote_trade(state, 0, shares)
                effective_entry = entry_quote.cost / shares if shares > 0 else price
            else:
                effective_entry = price
                shares = 1
                b = 0
                position_usd = price

            entry_time = row["timestamp"]
            outcome = None
            future_trades = market_trades[market_trades["timestamp"] > entry_time]
            for _, future in future_trades.iterrows():
                fp = bt._estimate_price(future)
                if fp >= 0.30:
                    outcome = {"exit": "TAKE_PROFIT", "raw_exit": fp}
                    break
                if fp <= 0.008:
                    outcome = {"exit": "STOP_LOSS", "raw_exit": fp}
                    break
            if outcome is None:
                res = market.get("answer1")
                outcome = {"exit": "RESOLUTION", "raw_exit": float(res) if res else 0}

            pnl = (outcome["raw_exit"] - effective_entry) / effective_entry * 100
            if mode == "lmsr":
                running_bankroll += (outcome["raw_exit"] - effective_entry) * shares
            results.append({
                "pnl_pct": pnl,
                "entry_price": price,
                "effective_entry": effective_entry,
                "exit_price": outcome["raw_exit"],
                "exit_type": outcome["exit"],
                "b_calibrated": b,
                "entry_slippage_bps": (effective_entry - price) / price * 10_000 if price > 0 else 0,
                "bankroll_after": running_bankroll,
            })
            break

    return results


if __name__ == "__main__":
    main()
