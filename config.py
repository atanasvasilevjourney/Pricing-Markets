"""
Centralized configuration — all tunable parameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API endpoints ──────────────────────────────────────────────────────
CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/polymarket-orderbook-v2/prod/gn"
)

# ── Wallet & auth ──────────────────────────────────────────────────────
POLYGON_WALLET_PRIVATE_KEY = os.getenv("POLYGON_WALLET_PRIVATE_KEY", "")
POLYMARKET_PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
POLYGON_RPC = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")
CHAIN_ID = 137  # Polygon mainnet

# ── LLM keys ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Trading parameters ─────────────────────────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "500"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "15"))

# ── Kill switch (production safety) ─────────────────────────────────────
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.10"))   # 10% max loss per day
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "20"))           # cap orders per calendar day
BALANCE_FILE = os.getenv("BALANCE_FILE", "")  # optional: path to file with current balance (one number)

# ── Henry-strategy optimal entry zone ──────────────────────────────────
OPTIMAL_ENTRY_MIN = 0.05
OPTIMAL_ENTRY_MAX = 0.20
MIN_VOLUME_FILTER = 5_000       # $5K+ 24h volume
MAX_VOLUME_FILTER = 100_000     # Below $100K (above = efficiently priced)
MIN_EDGE_THRESHOLD = 0.08       # 8 percentage points minimum edge
MIN_LIQUIDITY = 2_000           # $2K minimum to enter/exit cleanly

# ── Kelly & risk ───────────────────────────────────────────────────────
KELLY_FRACTION = 0.25           # Quarter Kelly
MAX_SINGLE_POSITION_PCT = 0.05  # 5% of bankroll per trade
MAX_TOTAL_EXPOSURE_PCT = 0.20   # 20% of bankroll total open
MAX_CORRELATED_CRYPTO = 2       # Max simultaneous crypto bets

# ── Position monitoring ────────────────────────────────────────────────
TAKE_PROFIT_PCT = 0.50          # Exit at +50% of entry
STOP_LOSS_PCT = 0.30            # Exit at -30% of entry
TIME_DECAY_DAYS = 3             # Auto-exit after 3 days without target

# ── Model paths ────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./model"))

# ── ML / LLM / LMSR composite weights ────────────────────────────────
# Three-source signal fusion (must sum to 1.0):
#   ML stacking ensemble   — fast, pattern-based
#   LLM consensus          — causal reasoning, news-aware
#   LMSR microstructure    — market-structure, impact-aware
ML_WEIGHT = 0.35
LLM_WEIGHT = 0.50
LMSR_WEIGHT = 0.15

# ── LMSR microstructure parameters ───────────────────────────────────
LMSR_FEE_RATE = 0.02               # Polymarket winner fee
LMSR_MIN_EDGE = 0.03               # Min |p_hat - p_mkt| for LMSR signal
LMSR_MIN_EDGE_AFTER_FEES = 0.01    # Min edge net of fees
LMSR_DEFAULT_B = 100.0             # Fallback when calibration fails
LMSR_MAX_IMPACT_PCT = 0.30         # Reject if impact > 30% of raw edge
LMSR_B_FLOOR = 5.0                 # Minimum b (very thin market)
LMSR_B_CEILING = 50_000.0          # Maximum b (deep market)
