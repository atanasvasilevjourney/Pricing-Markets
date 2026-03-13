"""
Multi-model LLM consensus for probability estimation.
Uses Claude + GPT-4o. Only trades when models AGREE (spread < 0.08).
Source: LiveTradeBench arXiv:2511.03628
"""

import json
import logging
import statistics
from typing import Optional, Dict

from .config import ANTHROPIC_API_KEY, OPENAI_API_KEY

log = logging.getLogger(__name__)


class LLMProbabilityEngine:

    def __init__(self):
        self._anthropic = None
        self._openai = None
        self._init_clients()

    def _init_clients(self):
        if ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic
                self._anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                log.warning("anthropic package not installed — Claude disabled")
        if OPENAI_API_KEY:
            try:
                import openai
                self._openai = openai.OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                log.warning("openai package not installed — GPT-4o disabled")

    def estimate_probability(
        self,
        question: str,
        current_price: float,
        market_metadata: dict,
        news_context: str = "",
    ) -> dict:
        prompt = self._build_prompt(
            question, current_price, market_metadata, news_context,
        )

        estimates: Dict[str, float] = {}

        if self._anthropic:
            try:
                est = self._query_claude(prompt)
                if est is not None:
                    estimates["claude"] = est
            except Exception as e:
                log.warning("Claude error: %s", e)

        if self._openai:
            try:
                est = self._query_gpt4o(prompt)
                if est is not None:
                    estimates["gpt4o"] = est
            except Exception as e:
                log.warning("GPT-4o error: %s", e)

        if not estimates:
            return {"fair_value": current_price, "confidence": 0, "spread": 1}

        values = list(estimates.values())
        consensus = statistics.mean(values)
        spread = max(values) - min(values)
        confidence = max(0, 1 - (spread / 0.20))

        return {
            "fair_value": consensus,
            "confidence": confidence,
            "spread": spread,
            "estimates": estimates,
            "agree": spread < 0.08,
        }

    # ── Prompt ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        question: str,
        current_price: float,
        metadata: dict,
        news: str,
    ) -> str:
        resolution_criteria = metadata.get("description", "Standard resolution")
        closes_at = metadata.get("endDate", "Unknown")
        volume = metadata.get("volume24hr", "Unknown")

        return f"""You are a calibrated probability forecaster for prediction markets.
Your job: estimate the TRUE probability of this event occurring.

MARKET QUESTION: {question}
CURRENT MARKET PRICE (implied probability): {current_price:.3f} ({current_price * 100:.1f}%)
RESOLUTION CRITERIA: {resolution_criteria}
CLOSES AT: {closes_at}
24H VOLUME: ${volume}

NEWS CONTEXT (most recent):
{news if news else "(No recent news provided — rely on base rates and reasoning)"}

INSTRUCTIONS:
1. Identify the BASE RATE for this type of event from historical precedent
2. Adjust for CURRENT CONTEXT — what specific factors push above or below base rate?
3. Assess CAUSAL IMPACT of any news (not surface correlation, but actual mechanism)
4. Consider WHO is likely trading this market and their potential biases
5. Account for RESOLUTION CRITERIA exactly as written
6. Compare your estimate to the CURRENT MARKET PRICE

CAUSAL FILTER: Before adjusting your probability for any news, explicitly state:
- What is the causal MECHANISM by which this news affects the outcome?
- On a scale of 1-10, how DIRECT is this causal link?
- If directness < 6, do NOT adjust your probability more than ±3pp for this news item.

Reply with ONLY a JSON object:
{{
  "base_rate": <float 0-1>,
  "current_context_adjustment": <float, positive or negative>,
  "fair_probability": <float 0-1>,
  "reasoning": "<one sentence justification>",
  "confidence": <float 0-1>
}}

Do NOT output any text outside the JSON."""

    # ── LLM calls ──────────────────────────────────────────────────────

    def _query_claude(self, prompt: str) -> Optional[float]:
        response = self._anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        data = json.loads(text)
        return float(data["fair_probability"])

    def _query_gpt4o(self, prompt: str) -> Optional[float]:
        response = self._openai.chat.completions.create(
            model="gpt-4o",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        data = json.loads(text)
        return float(data["fair_probability"])
