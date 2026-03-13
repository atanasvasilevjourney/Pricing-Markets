#!/usr/bin/env python3
"""
Polymarket Mispriced Event Detection & Trading System
Production Entry Point — v2.0

LEGAL: US persons are prohibited from trading on Polymarket.
Educational and research purposes only.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .config import BANKROLL, DRY_RUN, SCAN_INTERVAL_MINUTES, MODEL_DIR
from .fetcher import PolymarketFetcher
from .ml_engine import MLPredictionEngine
from .llm_engine import LLMProbabilityEngine
from .detector import MispricingDetector
from .signals import Signal, SignalClassifier
from .risk import RiskGate
from .arbitrage import CombinatorialArbScanner
from .execution import ExecutionEngine

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"trading_{datetime.now():%Y%m%d}.log"
        ),
    ],
)
log = logging.getLogger(__name__)


async def run_scan_cycle(
    fetcher: PolymarketFetcher,
    ml_engine: MLPredictionEngine,
    llm_engine: LLMProbabilityEngine,
    classifier: SignalClassifier,
    risk_gate: RiskGate,
    executor: ExecutionEngine | None,
    bankroll: float,
    dry_run: bool = True,
):
    log.info("═══ SCAN CYCLE START ═══")

    # 1. Detect mispriced markets
    detector = MispricingDetector(ml_engine, llm_engine, fetcher)
    opportunities = detector.scan_all_markets(
        n_markets=50, min_volume=5_000, min_edge=0.08,
    )
    log.info(
        "Found %d opportunities above min_edge threshold",
        len(opportunities),
    )

    # 2. Combinatorial arb scanner
    arb_scanner = CombinatorialArbScanner()
    arb_opps = []
    for market in fetcher.get_markets(limit=100):
        intra = arb_scanner.scan_intra_market(market, fetcher)
        if intra:
            arb_opps.append(intra)
            log.info(
                "ARB FOUND: %s | cost=%.3f | profit=%.2f%%",
                market.get("question", "?"),
                intra["total_cost"],
                intra["profit_pct"],
            )

    # 3. Classify signals
    decisions = []
    for opp in opportunities:
        decision = classifier.classify(opp, bankroll)
        if decision.signal != Signal.HOLD:
            approved, reason = risk_gate.check_all(decision)
            if approved:
                decisions.append(decision)
                log.info(
                    "SIGNAL %s: %s | edge=%.3f | size=$%.2f",
                    decision.signal.value,
                    decision.question[:60],
                    decision.composite_edge,
                    decision.position_size,
                )
            else:
                log.debug("Blocked: %s", reason)

    # 4. Execute (or log in dry-run)
    if dry_run:
        log.info(
            "DRY RUN — would place %d trades + %d arbs",
            len(decisions), len(arb_opps),
        )
        return decisions, arb_opps

    if executor is None:
        log.error("Executor is None in live mode — skipping")
        return decisions, arb_opps

    for decision in decisions:
        market = next(
            (o["market"] for o in opportunities
             if o["market_id"] == decision.market_id),
            None,
        )
        if not market:
            continue
        token_ids = market.get("clobTokenIds") or ["", ""]
        token_id = (
            token_ids[0] if decision.direction == "YES"
            else token_ids[1] if len(token_ids) > 1 else token_ids[0]
        )
        result = executor.execute_trade(decision, token_id)
        log.info("EXECUTED: %s", result)

    log.info("═══ SCAN CYCLE COMPLETE ═══")
    return decisions, arb_opps


async def main():
    bankroll = BANKROLL
    dry_run = DRY_RUN
    model_path = str(MODEL_DIR)

    log.info("Starting | Bankroll=$%.2f | DryRun=%s", bankroll, dry_run)

    fetcher = PolymarketFetcher()
    ml_engine = MLPredictionEngine()
    llm_engine = LLMProbabilityEngine()
    classifier = SignalClassifier()
    risk_gate = RiskGate(bankroll=bankroll)
    executor = ExecutionEngine() if not dry_run else None

    # Train or load model
    scaler_path = Path(model_path) / "scaler.pkl"
    if scaler_path.exists():
        log.info("Loading saved model…")
        ml_engine.load(model_path)
    else:
        log.info("Training model on live market data…")
        markets = fetcher.get_markets(limit=300, min_volume=1_000)
        metrics = ml_engine.train(markets, fetcher, n_training=150)
        log.info("Training metrics: %s", metrics)
        ml_engine.save(model_path)

    # Main loop
    while True:
        try:
            await run_scan_cycle(
                fetcher, ml_engine, llm_engine,
                classifier, risk_gate, executor,
                bankroll=bankroll,
                dry_run=dry_run,
            )
        except KeyboardInterrupt:
            log.info("Shutdown requested")
            break
        except Exception as e:
            log.error("Cycle error: %s", e, exc_info=True)

        log.info("Sleeping %d minutes…", SCAN_INTERVAL_MINUTES)
        await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)


def entry():
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    entry()
