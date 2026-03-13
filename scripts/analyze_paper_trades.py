#!/usr/bin/env python3
"""
Paper-trade analytics dashboard.

Reads daily CSV logs produced by paper_logger.py and generates
a full performance dashboard with matplotlib.

Usage:
    python scripts/analyze_paper_trades.py                    # latest log
    python scripts/analyze_paper_trades.py --file logs/paper_trades_20260313.csv
    python scripts/analyze_paper_trades.py --all              # merge all logs
    python scripts/analyze_paper_trades.py --demo             # generate demo data

Outputs (to output/ directory):
    paper_cumulative_pnl.png       — hypothetical cumulative PnL curve
    paper_signal_distribution.png  — signal type breakdown
    paper_edge_scatter.png         — composite edge vs position size
    paper_daily_pnl.png            — PnL histogram per day
    paper_lmsr_diagnostics.png     — b distribution, impact bps, capacity
    paper_rolling_winrate.png      — rolling win rate over time
    paper_position_sizes.png       — position size histogram
"""

import argparse
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")

COLORS = {
    "primary": "#4A90D9",
    "secondary": "#E5573F",
    "accent": "#2ECC71",
    "warn": "#F39C12",
    "bg": "#F8F9FA",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}

SIGNAL_COLORS = {
    "STRONG_BUY_YES": "#27AE60",
    "BUY_YES": "#82E0AA",
    "HOLD": "#BDC3C7",
    "BUY_NO": "#F5B7B1",
    "STRONG_BUY_NO": "#E74C3C",
}


def _style_ax(ax, title=""):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.4, color=COLORS["grid"], linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                      color=COLORS["text"], pad=10)


# ── Demo data generator ──────────────────────────────────────────────

def generate_demo_data(n_rows: int = 200, seed: int = 77) -> pd.DataFrame:
    """Generate realistic paper-trade CSV data for dashboard testing."""
    rng = np.random.default_rng(seed)
    signals = ["STRONG_BUY_YES", "BUY_YES", "BUY_NO", "STRONG_BUY_NO", "HOLD"]
    signal_weights = [0.15, 0.30, 0.10, 0.05, 0.40]

    rows = []
    base_time = datetime(2026, 3, 1, 8, 0, 0)

    for i in range(n_rows):
        ts = base_time + timedelta(minutes=i * 15 + int(rng.integers(0, 10)))
        sig = rng.choice(signals, p=signal_weights)
        direction = "YES" if "YES" in sig else ("NO" if "NO" in sig else "YES")
        price = round(rng.uniform(0.03, 0.85), 4)
        edge = round(rng.uniform(0.02, 0.25), 4) if sig != "HOLD" else round(rng.uniform(0, 0.02), 4)
        ml_edge = round(edge * rng.uniform(0.3, 0.7), 4)
        llm_edge = round(edge * rng.uniform(0.2, 0.8), 4)
        lmsr_edge = round(edge * rng.uniform(0.1, 0.5), 4)
        confidence = round(rng.uniform(0.4, 0.95), 4) if sig != "HOLD" else round(rng.uniform(0.1, 0.4), 4)
        b = round(rng.uniform(5, 500), 1)
        impact = round(rng.uniform(10, 3000), 0)
        capacity = round(rng.uniform(20, 5000), 2)
        size = round(rng.uniform(1, 25), 2) if sig != "HOLD" else 0.0
        p_true = round(np.clip(price + edge * (1 if direction == "YES" else -1), 0.01, 0.99), 4)

        rows.append({
            "timestamp": ts.isoformat(timespec="seconds"),
            "market_id": f"MKT-{rng.integers(1000, 9999)}",
            "question": f"Demo event {i} question?",
            "signal": sig,
            "direction": direction,
            "current_price": price,
            "p_true": p_true,
            "composite_edge": edge,
            "ml_edge": ml_edge,
            "llm_edge": llm_edge,
            "lmsr_edge": lmsr_edge,
            "lmsr_strength": rng.choice(["strong", "moderate", "weak", "none"]),
            "b_calibrated": b,
            "impact_bps": impact,
            "capacity_usd": capacity,
            "position_size": size,
            "confidence": confidence,
            "rationale": f"ML {ml_edge:.3f}, LLM {llm_edge:.3f}, LMSR {lmsr_edge:.3f}",
        })

    return pd.DataFrame(rows)


# ── Plotting functions ────────────────────────────────────────────────

def plot_cumulative_pnl(df: pd.DataFrame, path: Path):
    """
    Hypothetical cumulative PnL:
    For each non-HOLD signal, simulate PnL as edge * position_size * random_outcome.
    """
    trades = df[df["signal"] != "HOLD"].copy()
    if trades.empty:
        print("  No trades to plot PnL curve.")
        return

    rng = np.random.default_rng(42)
    edges = trades["composite_edge"].astype(float).values
    sizes = trades["position_size"].astype(float).values
    confs = trades["confidence"].astype(float).values

    win_probs = np.clip(0.5 + edges * 2, 0.3, 0.85)
    wins = rng.random(len(edges)) < win_probs
    pnls = np.where(wins, edges * sizes, -edges * sizes * 0.6)
    cum_pnl = np.cumsum(pnls)

    fig, ax = plt.subplots(figsize=(14, 6))
    _style_ax(ax, "Hypothetical Cumulative PnL ($)")

    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                    where=cum_pnl >= 0, alpha=0.3, color=COLORS["accent"])
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                    where=cum_pnl < 0, alpha=0.3, color=COLORS["secondary"])
    ax.plot(cum_pnl, color=COLORS["primary"], linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trade #", fontsize=11)
    ax.set_ylabel("Cumulative PnL ($)", fontsize=11)

    final = cum_pnl[-1]
    peak = np.max(cum_pnl)
    trough = np.min(cum_pnl)
    ax.annotate(f"Final: ${final:.2f}", xy=(len(cum_pnl) - 1, final),
                fontsize=11, fontweight="bold",
                color=COLORS["accent"] if final > 0 else COLORS["secondary"])

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_signal_distribution(df: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    counts = df["signal"].value_counts()
    colors = [SIGNAL_COLORS.get(s, "#BDC3C7") for s in counts.index]
    bars = ax1.barh(counts.index, counts.values, color=colors, edgecolor="white")
    _style_ax(ax1, "Signal Distribution")
    ax1.set_xlabel("Count")
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=11, fontweight="bold")

    trades = df[df["signal"] != "HOLD"]
    if not trades.empty:
        dir_counts = trades["direction"].value_counts()
        ax2.pie(dir_counts.values, labels=dir_counts.index,
                autopct="%1.0f%%",
                colors=[COLORS["accent"], COLORS["secondary"]],
                textprops={"fontsize": 12})
        ax2.set_title("Direction Split (trades only)", fontsize=13,
                      fontweight="bold", color=COLORS["text"])

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_edge_scatter(df: pd.DataFrame, path: Path):
    trades = df[df["signal"] != "HOLD"].copy()
    if trades.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    edges = trades["composite_edge"].astype(float)
    sizes = trades["position_size"].astype(float)
    confs = trades["confidence"].astype(float)

    sc = ax1.scatter(edges, sizes, c=confs, cmap="RdYlGn", s=40, alpha=0.7, edgecolors="white")
    _style_ax(ax1, "Edge vs Position Size (colored by confidence)")
    ax1.set_xlabel("Composite Edge")
    ax1.set_ylabel("Position Size ($)")
    fig.colorbar(sc, ax=ax1, label="Confidence", shrink=0.8)

    ml = trades["ml_edge"].astype(float)
    llm = trades["llm_edge"].astype(float)
    lmsr = trades["lmsr_edge"].astype(float)
    x = np.arange(3)
    means = [ml.mean(), llm.mean(), lmsr.mean()]
    stds = [ml.std(), llm.std(), lmsr.std()]
    bars = ax2.bar(x, means, yerr=stds, capsize=5,
                   color=[COLORS["primary"], COLORS["warn"], COLORS["secondary"]],
                   edgecolor="white", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["ML Edge", "LLM Edge", "LMSR Edge"])
    _style_ax(ax2, "Mean Edge by Source (± std)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_lmsr_diagnostics(df: pd.DataFrame, path: Path):
    trades = df[df["signal"] != "HOLD"].copy()
    if trades.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    b_vals = trades["b_calibrated"].astype(float)
    axes[0].hist(b_vals, bins=30, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    _style_ax(axes[0], "Calibrated b Distribution")
    axes[0].set_xlabel("b")
    axes[0].axvline(b_vals.median(), color=COLORS["text"], linestyle="--",
                    label=f"Median: {b_vals.median():.0f}")
    axes[0].legend()

    impact = trades["impact_bps"].astype(float)
    axes[1].hist(impact, bins=30, color=COLORS["secondary"], edgecolor="white", alpha=0.85)
    _style_ax(axes[1], "Impact (bps) Distribution")
    axes[1].set_xlabel("Slippage (bps)")
    axes[1].axvline(impact.median(), color=COLORS["text"], linestyle="--",
                    label=f"Median: {impact.median():.0f}")
    axes[1].legend()

    cap = trades["capacity_usd"].astype(float)
    sizes = trades["position_size"].astype(float)
    axes[2].scatter(cap, sizes, c=COLORS["accent"], alpha=0.6, edgecolors="white", s=30)
    _style_ax(axes[2], "Capacity vs Position Size")
    axes[2].set_xlabel("Market Capacity ($)")
    axes[2].set_ylabel("Position Size ($)")
    max_val = max(cap.max(), sizes.max()) * 1.1
    axes[2].plot([0, max_val], [0, max_val], color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rolling_winrate(df: pd.DataFrame, path: Path, window: int = 20):
    trades = df[df["signal"] != "HOLD"].copy()
    if len(trades) < window:
        print(f"  Not enough trades ({len(trades)}) for rolling window ({window})")
        return

    rng = np.random.default_rng(42)
    edges = trades["composite_edge"].astype(float).values
    win_probs = np.clip(0.5 + edges * 2, 0.3, 0.85)
    wins = (rng.random(len(edges)) < win_probs).astype(float)

    rolling = pd.Series(wins).rolling(window).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    _style_ax(ax, f"Rolling Win Rate (window={window} trades)")

    ax.plot(rolling, color=COLORS["primary"], linewidth=2)
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--", label="Break-even")
    ax.fill_between(range(len(rolling)), rolling, 0.5,
                    where=rolling >= 0.5, alpha=0.2, color=COLORS["accent"])
    ax.fill_between(range(len(rolling)), rolling, 0.5,
                    where=rolling < 0.5, alpha=0.2, color=COLORS["secondary"])
    ax.set_xlabel("Trade #", fontsize=11)
    ax.set_ylabel("Win Rate", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_position_sizes(df: pd.DataFrame, path: Path):
    trades = df[(df["signal"] != "HOLD") & (df["position_size"].astype(float) > 0)].copy()
    if trades.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = trades["position_size"].astype(float)
    ax1.hist(sizes, bins=25, color=COLORS["accent"], edgecolor="white", alpha=0.85)
    _style_ax(ax1, "Position Size Distribution")
    ax1.set_xlabel("Position Size ($)")
    ax1.set_ylabel("Frequency")
    ax1.axvline(sizes.mean(), color=COLORS["text"], linestyle="--",
                label=f"Mean: ${sizes.mean():.2f}")
    ax1.axvline(sizes.median(), color=COLORS["secondary"], linestyle="--",
                label=f"Median: ${sizes.median():.2f}")
    ax1.legend()

    strength_groups = trades.groupby("lmsr_strength")["position_size"].apply(
        lambda x: x.astype(float).mean()
    )
    if not strength_groups.empty:
        order = ["strong", "moderate", "weak", "none"]
        strength_groups = strength_groups.reindex([o for o in order if o in strength_groups.index])
        colors = [COLORS["accent"], COLORS["primary"], COLORS["warn"], COLORS["secondary"]]
        ax2.bar(strength_groups.index, strength_groups.values,
                color=colors[:len(strength_groups)], edgecolor="white")
        _style_ax(ax2, "Avg Position Size by LMSR Strength")
        ax2.set_ylabel("Avg Position Size ($)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper trade analytics")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific CSV file to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Merge all paper_trades_*.csv logs")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo data and analyze")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.demo:
        print("Generating demo paper-trade data...")
        df = generate_demo_data()
        demo_path = OUTPUT_DIR / "demo_paper_trades.csv"
        df.to_csv(demo_path, index=False)
        print(f"  Saved {demo_path} ({len(df)} rows)")
    elif args.all:
        files = sorted(glob.glob("logs/paper_trades_*.csv"))
        if not files:
            print("No paper_trades_*.csv files found in logs/. Use --demo.")
            sys.exit(1)
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Loaded {len(df)} rows from {len(files)} log files.")
    elif args.file:
        if not Path(args.file).exists():
            print(f"File not found: {args.file}")
            sys.exit(1)
        df = pd.read_csv(args.file)
        print(f"Loaded {len(df)} rows from {args.file}")
    else:
        files = sorted(glob.glob("logs/paper_trades_*.csv"))
        if files:
            df = pd.read_csv(files[-1])
            print(f"Loaded {len(df)} rows from {files[-1]}")
        else:
            print("No log files found. Run with --demo to generate test data.")
            sys.exit(1)

    # Stats
    n_total = len(df)
    n_trades = len(df[df["signal"] != "HOLD"])
    print("\n=== Paper Trade Summary ===")
    print(f"  Total signals: {n_total}")
    print(f"  Actionable trades: {n_trades} ({n_trades / n_total * 100:.0f}%)")
    print(f"  Signal breakdown: {df['signal'].value_counts().to_dict()}")

    if n_trades > 0:
        trades = df[df["signal"] != "HOLD"]
        print(f"  Avg edge: {trades['composite_edge'].astype(float).mean():.4f}")
        print(f"  Avg position: ${trades['position_size'].astype(float).mean():.2f}")
        print(f"  Avg impact: {trades['impact_bps'].astype(float).mean():.0f} bps")

    # Generate all charts
    print("\n=== Generating Dashboard ===")
    plot_cumulative_pnl(df, OUTPUT_DIR / "paper_cumulative_pnl.png")
    plot_signal_distribution(df, OUTPUT_DIR / "paper_signal_distribution.png")
    plot_edge_scatter(df, OUTPUT_DIR / "paper_edge_scatter.png")
    plot_lmsr_diagnostics(df, OUTPUT_DIR / "paper_lmsr_diagnostics.png")
    plot_rolling_winrate(df, OUTPUT_DIR / "paper_rolling_winrate.png")
    plot_position_sizes(df, OUTPUT_DIR / "paper_position_sizes.png")

    print(f"\nAll charts saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
