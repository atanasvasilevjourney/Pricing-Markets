#!/usr/bin/env python3
"""
Polymarket Mispriced Event Detection & Trading System
Production Entry Point — v3.0 (ML + LLM + LMSR microstructure)

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
from .lmsr_features import LMSRAdapter
from .detector import MispricingDetector
from .signals import Signal, SignalClassifier
from .risk import RiskGate
from .arbitrage import CombinatorialArbScanner
from .execution import ExecutionEngine
from .paper_logger import PaperTradeLogger

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
    lmsr_adapter: LMSRAdapter,
    classifier: SignalClassifier,
    risk_gate: RiskGate,
    executor: ExecutionEngine | None,
    paper_logger: PaperTradeLogger,
    bankroll: float,
    dry_run: bool = True,
):
    log.info("═══ SCAN CYCLE START (v3 — ML+LLM+LMSR) ═══")

    # 1. Three-source mispricing detection
    detector = MispricingDetector(
        ml_engine, llm_engine, fetcher, lmsr_adapter,
    )
    opportunities = detector.scan_all_markets(
        n_markets=50, min_volume=5_000, min_edge=0.08,
    )
    log.info(
        "Found %d opportunities above min_edge threshold",
        len(opportunities),
    )

    for opp in opportunities[:5]:
        lmsr = opp.get("lmsr_analysis") or {}
        log.info(
            "  → %s | composite=%.3f | ML=%.3f LLM=%.3f LMSR=%.3f "
            "| b=%.0f | impact=%.0fbps",
            opp["question"][:55],
            opp["composite_edge"],
            opp["ml_edge"],
            opp.get("llm_edge", 0),
            opp.get("lmsr_edge", 0),
            lmsr.get("b_calibrated", 0),
            (lmsr.get("impact_at_ref_size") or {}).get("slippage_bps", 0),
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

    # 3. Classify signals (with impact-adjusted Kelly sizing)
    decisions = []
    for opp in opportunities:
        decision = classifier.classify(opp, bankroll)
        if decision.signal != Signal.HOLD:
            approved, reason = risk_gate.check_all(decision)
            if approved:
                decisions.append(decision)
                paper_logger.log_decision(decision, opp)
                log.info(
                    "SIGNAL %s: %s | edge=%.3f | size=$%.2f "
                    "| lmsr=%s | impact=%.0fbps",
                    decision.signal.value,
                    decision.question[:50],
                    decision.composite_edge,
                    decision.position_size,
                    decision.lmsr_strength,
                    decision.impact_bps,
                )
            else:
                log.debug("Blocked: %s", reason)

    for arb in arb_opps:
        paper_logger.log_arb(arb)

    # 4. Execute (or log in dry-run)
    log.info(paper_logger.summary())

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

    log.info(
        "Starting v3 (ML+LLM+LMSR) | Bankroll=$%.2f | DryRun=%s",
        bankroll, dry_run,
    )

    fetcher = PolymarketFetcher()
    ml_engine = MLPredictionEngine()
    llm_engine = LLMProbabilityEngine()
    lmsr_adapter = LMSRAdapter()
    classifier = SignalClassifier(lmsr_adapter=lmsr_adapter)
    risk_gate = RiskGate(bankroll=bankroll)
    executor = ExecutionEngine() if not dry_run else None
    paper_logger = PaperTradeLogger()

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
                fetcher, ml_engine, llm_engine, lmsr_adapter,
                classifier, risk_gate, executor, paper_logger,
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
