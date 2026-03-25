"""Rubric evaluator with rule-based scoring and optional async LLM judge."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from pokemonred_puffer.rubric_rl.rubrics import (
    CompositeRubric,
    CompositeRubricResult,
    GameStateSnapshot,
)
from pokemonred_puffer.rubric_rl.state_summarizer import StateSummarizer

logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeConfig:
    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key: str | None = None
    max_workers: int = 4
    blend_weight: float = 0.0  # 0 = rule-based only, 1 = LLM only


class RubricEvaluator:
    """Evaluates game states against rubrics.

    Supports two modes:
    1. Rule-based (default): Fast, synchronous evaluation using criterion functions
    2. LLM judge (optional): Async LLM evaluation for richer feedback
    """

    def __init__(
        self,
        rubric: CompositeRubric,
        llm_config: LLMJudgeConfig | None = None,
    ):
        self.rubric = rubric
        self.summarizer = StateSummarizer()
        self.llm_config = llm_config or LLMJudgeConfig()
        self._llm_client = None
        self._executor = None

        if self.llm_config.enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=self.llm_config.max_workers
            )

    def evaluate(self, snapshot: GameStateSnapshot) -> CompositeRubricResult:
        """Synchronous rule-based evaluation."""
        return self.rubric.score(snapshot)

    def evaluate_batch(
        self, snapshots: list[GameStateSnapshot]
    ) -> list[CompositeRubricResult]:
        """Batch rule-based evaluation."""
        return [self.rubric.score(s) for s in snapshots]

    def evaluate_with_llm_async(
        self, snapshot: GameStateSnapshot
    ) -> Future[float] | None:
        """Submit async LLM evaluation. Returns a Future or None if LLM is disabled."""
        if not self.llm_config.enabled or self._executor is None:
            return None

        summary = self.summarizer.summarize(snapshot)
        return self._executor.submit(self._llm_evaluate, summary)

    def _llm_evaluate(self, summary: str) -> float:
        """Call LLM to evaluate a game state summary. Returns score in [0, 1]."""
        try:
            client = self._get_llm_client()
            if client is None:
                return 0.0

            prompt = self._build_evaluation_prompt(summary)

            if self.llm_config.provider == "anthropic":
                response = client.messages.create(
                    model=self.llm_config.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                text = response.choices[0].message.content

            return self._parse_score(text)
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return 0.0

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        try:
            if self.llm_config.provider == "anthropic":
                import anthropic

                self._llm_client = anthropic.Anthropic(
                    api_key=self.llm_config.api_key
                )
            else:
                import openai

                self._llm_client = openai.OpenAI(api_key=self.llm_config.api_key)
        except ImportError:
            logger.warning(
                f"LLM provider {self.llm_config.provider} not installed"
            )
            self.llm_config.enabled = False
            return None

        return self._llm_client

    def _build_evaluation_prompt(self, summary: str) -> str:
        criteria_text = []
        for _, rubric in self.rubric.rubrics:
            for c in rubric.criteria:
                criteria_text.append(f"- {c.name} (weight {c.weight}): {c.description}")

        return f"""You are evaluating a Pokemon Red gameplay session. Rate the overall quality of play on a scale from 0.0 to 1.0.

## Evaluation Criteria
{chr(10).join(criteria_text)}

## Game Session Summary
{summary}

Respond with ONLY a single number between 0.0 and 1.0 representing the overall score."""

    def _parse_score(self, text: str) -> float:
        """Extract a float score from LLM response text."""
        text = text.strip()
        try:
            score = float(text)
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to find a number in the text
            import re

            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.0

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
