#!/usr/bin/env python3
"""
Fetch real Polymarket data for different markets and save in backtest-ready form.

Requires outbound HTTPS to gamma-api.polymarket.com and clob.polymarket.com.
If you are offline or blocked, use --demo to copy synthetic data into data/ instead.

Uses the project's PolymarketFetcher to pull:
  - Markets from Gamma API (multiple orderings and active/inactive)
  - Recent CLOB trades per market (YES token)
  - Price history per market (converted to trade-like rows for more data)

Outputs (default data/):
  - data/trades.csv   : timestamp, makerAssetId (condition_id), makerAmountFilled, takerAmountFilled
  - data/markets.csv  : condition_id, createdAt, closedTime, volume, answer1, question

Usage:
  python scripts/fetch_polymarket_data.py
  python scripts/fetch_polymarket_data.py --out-dir data --max-markets 80
  python scripts/fetch_polymarket_data.py --demo   # copy synthetic data when offline
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from Polymarket.fetcher import PolymarketFetcher

# Backtester expects: makerAssetId = condition_id, price = makerAmountFilled / takerAmountFilled
FILL_SCALE = 100.0  # store price as (p * FILL_SCALE) / FILL_SCALE so price = p


def _condition_id(m: dict) -> str | None:
    """Prefer conditionId, fall back to id."""
    cid = m.get("conditionId") or m.get("condition_id") or m.get("id")
    return str(cid) if cid else None


def _yes_token(m: dict) -> str | None:
    tokens = m.get("clobTokenIds") or m.get("tokens") or []
    if isinstance(tokens, list) and len(tokens) > 0:
        return str(tokens[0])
    return None


def _parse_ts(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            dt = datetime.fromtimestamp(int(v) / 1000.0 if v > 1e12 else v, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    if isinstance(v, str):
        return v[:19].replace("T", " ")
    return None


def trade_row_from_clob(t: dict, condition_id: str) -> dict | None:
    """
    Convert one CLOB trade to backtest row.
    Backtester price = makerAmountFilled / takerAmountFilled.
    """
    ts = t.get("timestamp") or t.get("t") or t.get("createdAt")
    ts_str = _parse_ts(ts)
    if not ts_str:
        return None

    # CLOB may return makerAssetId, makerAmountFilled, takerAmountFilled
    maker = float(t.get("makerAmountFilled", 0) or 0)
    taker = float(t.get("takerAmountFilled", 0) or 0)
    if maker > 0 and taker > 0:
        # Price = USDC / shares. Assume one side is USDC, one is outcome.
        # If maker is outcome: price = taker/maker. If maker is USDC: price = maker/taker.
        # Backtester: price = maker_amt / taker_amt. So we want maker = price*K, taker = K.
        # So set makerAmountFilled = price * FILL_SCALE, takerAmountFilled = FILL_SCALE.
        maker_asset = str(t.get("makerAssetId", "")).upper()
        if "USDC" in maker_asset:
            price = maker / taker
        else:
            price = taker / maker
        price = max(0.001, min(0.999, price))
        return {
            "timestamp": ts_str,
            "makerAssetId": condition_id,
            "makerAmountFilled": round(price * FILL_SCALE, 4),
            "takerAmountFilled": FILL_SCALE,
        }

    # Fallback: price and size
    price = float(t.get("price", 0) or 0)
    size = float(t.get("size", 0) or t.get("amount", 0) or 1)
    if price <= 0 or size <= 0:
        return None
    price = max(0.001, min(0.999, price))
    return {
        "timestamp": ts_str,
        "makerAssetId": condition_id,
        "makerAmountFilled": round(price * FILL_SCALE, 4),
        "takerAmountFilled": FILL_SCALE,
    }


def trade_rows_from_price_history(history: list, condition_id: str) -> list[dict]:
    """Turn price history into one row per point so we have more time series for backtest."""
    rows = []
    if not history:
        return rows
    for i, point in enumerate(history):
        if isinstance(point, dict):
            ts = point.get("t") or point.get("timestamp") or point.get("date")
            p = point.get("p") or point.get("price")
        else:
            ts, p = None, None
        if p is None:
            continue
        try:
            price = float(p)
        except (TypeError, ValueError):
            continue
        price = max(0.001, min(0.999, price))
        ts_str = _parse_ts(ts)
        if not ts_str:
            ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "timestamp": ts_str,
            "makerAssetId": condition_id,
            "makerAmountFilled": round(price * FILL_SCALE, 4),
            "takerAmountFilled": FILL_SCALE,
        })
    return rows


def market_row(m: dict) -> dict:
    """One market row for markets.csv (backtester + run_real_backtest format)."""
    cid = _condition_id(m)
    if not cid:
        return None

    created = m.get("createdAt") or m.get("creationDate") or m.get("startDate") or m.get("created")
    closed = m.get("closedTime") or m.get("endDate") or m.get("closed") or m.get("end_date")
    if isinstance(created, (int, float)):
        created = datetime.fromtimestamp(created / 1000.0 if created > 1e12 else created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(closed, (int, float)):
        closed = datetime.fromtimestamp(closed / 1000.0 if closed > 1e12 else closed, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if not created:
        created = "2020-01-01 00:00:00"
    if not closed:
        closed = "2030-12-31 23:59:59"

    vol = float(m.get("volume", 0) or m.get("volume24hr", 0) or m.get("volumeNum", 0) or 0)
    # Resolved outcome: outcomePrices might be [1,0] or [0,1], or outcome, resolution
    answer1 = 0.5
    out = m.get("outcomePrices") or m.get("prices") or []
    if isinstance(out, list) and len(out) >= 1:
        try:
            answer1 = float(out[0])
        except (TypeError, ValueError):
            pass
    if "resolution" in m:
        r = m["resolution"]
        if r in ("YES", "yes", "1"):
            answer1 = 1.0
        elif r in ("NO", "no", "0"):
            answer1 = 0.0
    if "outcome" in m and m["outcome"] in ("YES", "yes"):
        answer1 = 1.0
    elif "outcome" in m and m["outcome"] in ("NO", "no"):
        answer1 = 0.0

    return {
        "condition_id": cid,
        "createdAt": created,
        "closedTime": closed,
        "volume": vol,
        "answer1": answer1,
        "question": str(m.get("question", ""))[:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch real Polymarket data for backtesting")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory (default: data)")
    parser.add_argument("--max-markets", type=int, default=120, help="Max distinct markets to fetch")
    parser.add_argument("--trades-per-market", type=int, default=100, help="Max CLOB trades per market")
    parser.add_argument("--skip-price-history", action="store_true", help="Do not add price-history as trade rows")
    parser.add_argument("--demo", action="store_true", help="When offline: copy output/synth_*.csv to out-dir if present")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        for name in ("synth_trades", "synth_markets"):
            src = REPO_ROOT / "output" / f"{name}.csv"
            if src.exists():
                dst = out_dir / ("trades.csv" if "trades" in name else "markets.csv")
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")
        print("Demo: run backtest with --trades data/trades.csv --markets data/markets.csv")
        return

    fetcher = PolymarketFetcher(rate_limit_delay=0.6)

    # Collect markets from multiple orderings to get variety
    seen_cids = set()
    all_markets = []

    for order in ("volume24hr", "liquidity", "volume"):
        raw = fetcher.get_markets(limit=80, order=order, min_volume=1_000, active_only=True)
        for m in raw:
            cid = _condition_id(m)
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                all_markets.append(m)
        if len(all_markets) >= args.max_markets:
            break

    # Also get some inactive (resolved) if API returns them
    raw_inactive = fetcher.get_markets(limit=40, order="volume24hr", min_volume=500, active_only=False)
    for m in raw_inactive:
        cid = _condition_id(m)
        if cid and cid not in seen_cids:
            seen_cids.add(cid)
            all_markets.append(m)
        if len(all_markets) >= args.max_markets:
            break

    all_markets = all_markets[: args.max_markets]
    if not all_markets:
        print("No markets returned. Check network (gamma-api.polymarket.com). Use --demo to copy synthetic data.")
        sys.exit(1)
    print(f"Collected {len(all_markets)} distinct markets")

    # Build markets.csv
    market_rows = []
    for m in all_markets:
        row = market_row(m)
        if row:
            market_rows.append(row)

    markets_df = pd.DataFrame(market_rows)
    markets_path = out_dir / "markets.csv"
    markets_df.to_csv(markets_path, index=False)
    print(f"Saved {len(market_rows)} markets to {markets_path}")

    # Build trades from CLOB + optional price history
    trade_rows = []
    for i, m in enumerate(all_markets):
        cid = _condition_id(m)
        token = _yes_token(m)
        if not cid or not token:
            continue

        # CLOB trades
        trades = fetcher.get_recent_trades(token, limit=args.trades_per_market)
        for t in trades:
            row = trade_row_from_clob(t, cid)
            if row:
                trade_rows.append(row)

        # Price history as trade-like rows (more data for backtest)
        if not args.skip_price_history:
            history = fetcher.get_price_history(token, interval="1h", fidelity=170)
            for row in trade_rows_from_price_history(history or [], cid):
                trade_rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  Fetched trades/history for {i + 1}/{len(all_markets)} markets ...")

    if not trade_rows:
        print("WARNING: No trades collected. Check API access and token/condition ids.")
        sys.exit(1)

    trades_df = pd.DataFrame(trade_rows)
    # Sort by time for backtester
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")
    trades_df = trades_df.dropna(subset=["timestamp"])
    trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
    trades_df["timestamp"] = trades_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trades_path = out_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved {len(trades_df)} trade rows to {trades_path}")

    print("\nDone. Run backtest with:")
    print(f"  python scripts/run_real_backtest.py --trades {trades_path} --markets {markets_path}")
    print(f"  python scripts/run_backtests.py --trades {trades_path} --markets {markets_path}")


if __name__ == "__main__":
    main()
