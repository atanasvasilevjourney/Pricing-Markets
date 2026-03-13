#!/usr/bin/env python3
"""
Top-level runner — use this from outside the package:

    python polymarket_bot/run.py          # from FINSOPS/
    python -m polymarket_bot              # alternative (package runner)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_bot.main import entry

if __name__ == "__main__":
    entry()
