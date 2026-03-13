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

# ── ML composite weights ──────────────────────────────────────────────
ML_WEIGHT = 0.4
LLM_WEIGHT = 0.6
