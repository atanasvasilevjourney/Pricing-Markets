#!/usr/bin/env python3
"""
Real-data backtest loader for poly_data (warproxxx/poly_data) CSVs.

Validates column shapes, runs both naive Henry and LMSR-aware strategies,
and writes a one-page summary to output/backtest_summary.txt.

Usage:
    python scripts/run_real_backtest.py --trades data/trades.csv --markets data/markets.csv
    python scripts/run_real_backtest.py --trades path/to/processed/trades.csv --markets path/to/markets.csv

Expected CSV shapes (poly_data):
  trades:  timestamp, makerAssetId, makerAmountFilled, takerAmountFilled
           (makerAssetId identifies the market/condition)
  markets: condition_id, createdAt, closedTime, volume, answer1
           (answer1 = 1.0 if YES won, 0.0 if NO won)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from Polymarket.backtester import Backtester

OUTPUT_DIR = Path("output")

# Required columns (backtester and our logic)
TRADES_REQUIRED = {"timestamp", "makerAssetId", "makerAmountFilled", "takerAmountFilled"}
MARKETS_REQUIRED = {"condition_id", "createdAt", "closedTime", "answer1"}
# Optional but used by LMSR-aware: volume
MARKETS_OPTIONAL = {"volume", "question"}


# Canonical names expected by Backtester
TRADES_COLS = ["timestamp", "makerAssetId", "makerAmountFilled", "takerAmountFilled"]
MARKETS_COLS = ["condition_id", "createdAt", "closedTime", "answer1", "volume"]


def normalize_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Lowercase for validation; then map to Backtester's expected names."""
    df = df.copy()
    raw = {c.strip().lower(): c for c in df.columns}
    df.columns = [c.strip().lower() for c in df.columns]

    if name == "trades":
        if "timestamp" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        # Map to canonical
        for want in TRADES_COLS:
            w = want.lower()
            if w in df.columns and df.columns[df.columns == w].tolist():
                df = df.rename(columns={w: want})
        return df

    if name == "markets":
        if "condition_id" not in df.columns and "market_id" in df.columns:
            df = df.rename(columns={"market_id": "condition_id"})
        for want in ["createdAt", "closedTime", "condition_id", "answer1", "volume"]:
            w = want.lower()
            if w in df.columns:
                df = df.rename(columns={w: want})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df

    return df


def validate_trades(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Load trades CSV and validate. Returns (df, errors)."""
    errors = []
    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as e:
        return None, [f"Could not read trades file: {e}"]

    df = pd.read_csv(path)
    df = normalize_columns(df, "trades")

    missing = TRADES_REQUIRED - set(df.columns)
    if missing:
        errors.append(f"Trades missing required columns: {missing}")
        return None, errors

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception as e:
        errors.append(f"Trades 'timestamp' not parseable: {e}")
        return None, errors

    if len(df) == 0:
        errors.append("Trades file is empty")
        return None, errors

    return df, []


def validate_markets(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    """Load markets CSV and validate. Returns (df, errors)."""
    errors = []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, [f"Could not read markets file: {e}"]

    df = normalize_columns(df, "markets")

    missing = MARKETS_REQUIRED - set(df.columns)
    if missing:
        errors.append(f"Markets missing required columns: {missing}")
        return None, errors

    for col in ["createdAt", "closedTime"]:
        if col not in df.columns and col.lower() in df.columns:
            df = df.rename(columns={col.lower(): col})
    try:
        df["createdAt"] = pd.to_datetime(df["createdAt"])
        df["closedTime"] = pd.to_datetime(df["closedTime"])
    except Exception as e:
        errors.append(f"Markets date columns not parseable: {e}")
        return None, errors

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if len(df) == 0:
        errors.append("Markets file is empty")
        return None, errors

    return df, []


def run_backtests(trades_path: Path, markets_path: Path, bankroll: float = 500.0) -> dict:
    """Run both strategies and return raw results plus summary text."""
    bt = Backtester(str(trades_path), str(markets_path))

    naive = bt.backtest_henry_strategy(
        entry_min_price=0.02,
        entry_max_price=0.20,
        target_price=0.30,
        stop_loss=0.008,
    )
    lmsr = bt.backtest_lmsr_aware(
        entry_min_price=0.02,
        entry_max_price=0.20,
        target_price=0.30,
        stop_loss=0.008,
        bankroll=bankroll,
    )

    return {
        "naive": naive,
        "lmsr": lmsr,
        "n_trades_raw": len(bt.trades),
        "n_markets": len(bt.markets),
    }


def write_summary(
    trades_path: Path,
    markets_path: Path,
    results: dict,
    bankroll: float,
    out_path: Path,
) -> None:
    """Write one-page backtest_summary.txt."""
    n = results["naive"]
    l_ = results["lmsr"]
    lines = [
        "=" * 70,
        "BACKTEST SUMMARY (real data)",
        "=" * 70,
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Trades file: {trades_path}",
        f"Markets file: {markets_path}",
        f"Bankroll (LMSR): ${bankroll:.2f}",
        "",
        f"Data: {results['n_trades_raw']:,} trade rows, {results['n_markets']} markets",
        "",
        "-" * 70,
        "NAIVE HENRY STRATEGY",
        "-" * 70,
    ]

    if "error" in n:
        lines.append(f"  Error: {n['error']}")
    else:
        lines.extend([
            f"  Trades:        {n.get('n_trades', 0)}",
            f"  Win rate:      {n.get('win_rate', 0):.1f}%",
            f"  Avg PnL:       {n.get('avg_pnl_pct', 0):.2f}%",
            f"  Median PnL:    {n.get('median_pnl_pct', 0):.2f}%",
            f"  Max win:       {n.get('max_win', 0):.2f}%",
            f"  Max loss:      {n.get('max_loss', 0):.2f}%",
            f"  Total return:  {n.get('total_return', 0):.2f}%",
            f"  Sharpe (approx): {n.get('sharpe_approx', 0):.3f}",
            f"  Exit breakdown: {n.get('exit_breakdown', {})}",
        ])

    lines.extend([
        "",
        "-" * 70,
        "LMSR-AWARE STRATEGY",
        "-" * 70,
    ])

    if "error" in l_:
        lines.append(f"  Error: {l_['error']}")
    else:
        lines.extend([
            f"  Trades:        {l_.get('n_trades', 0)}",
            f"  Win rate:      {l_.get('win_rate', 0):.1f}%",
            f"  Avg PnL:       {l_.get('avg_pnl_pct', 0):.2f}%",
            f"  Median PnL:    {l_.get('median_pnl_pct', 0):.2f}%",
            f"  Total return:  {l_.get('total_return_pct', 0):.2f}%",
            f"  Final bankroll: ${l_.get('final_bankroll', 0):.2f}",
            f"  Max drawdown:  {l_.get('max_drawdown_pct', 0):.1f}%",
            f"  Avg slippage:  {l_.get('avg_entry_slippage_bps', 0):.0f} bps",
            f"  Sharpe (approx): {l_.get('sharpe_approx', 0):.3f}",
            f"  Exit breakdown: {l_.get('exit_breakdown', {})}",
        ])

    lines.extend([
        "",
        "-" * 70,
        "COMPARISON",
        "-" * 70,
    ])
    if "error" not in n and "error" not in l_:
        lines.append(
            f"  Sharpe delta (LMSR - Naive): {l_.get('sharpe_approx', 0) - n.get('sharpe_approx', 0):+.3f}"
        )
        lines.append(
            f"  Win rate delta: {l_.get('win_rate', 0) - n.get('win_rate', 0):+.1f} pp"
        )
    lines.append("")
    lines.append("=" * 70)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run real-data backtest (poly_data CSV shape)"
    )
    parser.add_argument("--trades", type=str, required=True, help="Path to trades.csv")
    parser.add_argument("--markets", type=str, required=True, help="Path to markets.csv")
    parser.add_argument("--bankroll", type=float, default=500.0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for summary (default: output/backtest_summary.txt)")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    markets_path = Path(args.markets)

    if not trades_path.exists():
        print(f"ERROR: Trades file not found: {trades_path}")
        sys.exit(1)
    if not markets_path.exists():
        print(f"ERROR: Markets file not found: {markets_path}")
        sys.exit(1)

    print("Validating trades CSV...")
    trades_df, errs = validate_trades(trades_path)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
        print("Required columns:", TRADES_REQUIRED)
        sys.exit(1)
    print(f"  OK: {len(trades_df):,} rows")

    print("Validating markets CSV...")
    markets_df, errs = validate_markets(markets_path)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
        print("Required columns:", MARKETS_REQUIRED)
        sys.exit(1)
    print(f"  OK: {len(markets_df):,} rows")

    # Write normalized CSVs to a temp dir so Backtester gets the right column names
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        t_path = Path(tmp) / "trades.csv"
        m_path = Path(tmp) / "markets.csv"
        trades_df.to_csv(t_path, index=False)
        markets_df.to_csv(m_path, index=False)

        print("\nRunning naive Henry strategy...")
        print("Running LMSR-aware strategy...")
        results = run_backtests(t_path, m_path, bankroll=args.bankroll)

    out_path = Path(args.output) if args.output else OUTPUT_DIR / "backtest_summary.txt"
    write_summary(trades_path, markets_path, results, args.bankroll, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
