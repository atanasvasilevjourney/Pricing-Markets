# Polymarket Mispriced Event Detection & Trading System

**Version 2.0** — Production architecture for detecting and trading mispriced prediction-market contracts on [Polymarket](https://polymarket.com).

> **Legal:** US persons are prohibited from trading on Polymarket per its Terms of Service. This project is for **educational and research purposes only**. Not financial advice.

---

## What It Does

The system finds **mispriced** contracts: when the market’s implied probability diverges from a more accurate estimate. It targets two types:

1. **Probabilistic mispricing (Henry strategy)** — Buy underpriced YES, wait for the market to reprice, then sell before resolution. You profit from the crowd updating, not from the event happening.
2. **Mathematical mispricing (arbitrage)** — When YES + NO (or all outcomes) sum to less than $1.00, buying all sides locks in a profit at resolution.

The pipeline: **fetch markets → extract 10 features → ML + LLM consensus → signal classification → Kelly sizing → risk checks → CLOB execution** (optional).

---

## Requirements

- **Python 3.10+**
- **APIs:** Polymarket (public), optional: Anthropic + OpenAI for LLM consensus
- **Live trading:** Polygon wallet with USDC, `py-clob-client` and private key (see [Polymarket docs](https://docs.polymarket.com))

---

## Installation

```bash
cd polymarket_bot
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

---

## Configuration

1. Copy the env template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` (never commit it):

   | Variable | Description |
   |----------|-------------|
   | `POLYGON_WALLET_PRIVATE_KEY` | Required for live trading |
   | `POLYMARKET_PROXY_ADDRESS`   | Safe proxy (optional) |
   | `ANTHROPIC_API_KEY`          | Optional — Claude probability estimates |
   | `OPENAI_API_KEY`             | Optional — GPT-4o probability estimates |
   | `BANKROLL`                  | Simulated/real bankroll in USD (default `500`) |
   | `DRY_RUN`                   | `true` = no real orders (default) |
   | `SCAN_INTERVAL_MINUTES`     | Minutes between scans (default `15`) |

With no LLM keys, the bot still runs using ML-only edge (LLM weight is skipped).

---

## Usage

**Dry run (no real orders, recommended first):**
```bash
DRY_RUN=true python -m polymarket_bot
```

**From repo root (FINSOPS):**
```bash
python polymarket_bot/run.py
```

**Live trading (only after testing):**
```bash
DRY_RUN=false python -m polymarket_bot
```

Logs go to the console and to `logs/trading_YYYYMMDD.log`.

---

## Project Structure

```
polymarket_bot/
├── README.md           # This file
├── requirements.txt
├── .env.example        # Copy to .env and fill in
├── config.py           # Central config (env + constants)
├── fetcher.py          # CLOB, Gamma, Data APIs + Goldsky
├── features.py         # 10-feature extractor
├── ml_engine.py        # 5-model stacking ensemble (XGB, LGB, etc.)
├── llm_engine.py       # Claude + GPT-4o probability consensus
├── detector.py         # Mispricing detection + Henry filter
├── kelly.py            # Quarter-Kelly position sizing
├── signals.py          # Signal classifier + TradeDecision
├── risk.py             # RiskGate + PositionMonitor
├── arbitrage.py        # Combinatorial arb scanner
├── execution.py        # CLOB order placement
├── backtester.py       # Walk-forward backtest on historical data
├── main.py             # Main loop and scan cycle
├── run.py              # Top-level script entry
├── logs/               # Created at runtime
└── model/              # Saved ML models (created on first train)
```

---

## Quick Start Checklist

- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Copy env: `cp .env.example .env`
- [ ] Set `BANKROLL`, optionally LLM and wallet keys in `.env`
- [ ] Run in dry run: `DRY_RUN=true python -m polymarket_bot`
- [ ] Watch logs for 1–2 weeks to validate signals
- [ ] (Optional) Run backtest with `backtester.py` if you have historical CSVs
- [ ] Only then consider `DRY_RUN=false` with a small bankroll

---

## Backtesting

`backtester.py` expects historical data in the format used by [warproxxx/poly_data](https://github.com/warproxxx/poly_data) (e.g. `trades.csv`, `markets.csv`). Example:

```python
from polymarket_bot.backtester import Backtester

bt = Backtester("path/to/trades.csv", "path/to/markets.csv")
results = bt.backtest_henry_strategy(
    entry_min_price=0.02,
    entry_max_price=0.20,
    target_price=0.30,
    stop_loss=0.008,
)
print(results)  # win_rate, avg_pnl_pct, exit_breakdown, etc.
```

---

## Risk and Defaults

- **Quarter Kelly** with confidence scaling; hard cap **5% of bankroll per trade**, **20% total exposure**.
- **Correlation:** limits multiple correlated positions (e.g. crypto).
- **Liquidity:** ignores markets below configured liquidity so you can exit cleanly.

Tune these in `config.py` or via env where supported.

---

## References

- [Polymarket API docs](https://docs.polymarket.com)
- NavnoorBawa/polymarket-prediction-system (GitHub)
- arXiv:2508.03474 (AFT 2025 — combinatorial arb)
- arXiv:2511.03628 (LiveTradeBench — LLM forecasting)
- warproxxx/poly_data (historical data)

---

## License

Use for education and research only. Check Polymarket’s Terms of Service and local laws before any real trading.
