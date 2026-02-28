"""
Task C — LLM Self-Coherence Judgment.

Ask the LLM to rate the coherence of the (possibly corrupted) text on a
1–10 scale and explain its reasoning.  This is a meta-task: we want to know
whether the LLM is *aware* that the text has become incoherent.

The parsed output extracts the numeric score so it can be correlated with
objective task-performance metrics downstream.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .base import ProbingTask, ProbingResult

_SYSTEM_PROMPT = (
    "You are a text quality evaluator. "
    "Rate the coherence and meaningfulness of the provided text on a scale from 1 to 10, "
    "where 1 = completely incoherent / random noise and 10 = perfectly clear and meaningful. "
    "Respond in exactly this format:\n"
    "SCORE: <integer 1-10>\n"
    "REASON: <one sentence explanation>"
)

_USER_TEMPLATE = """\
TEXT TO EVALUATE:
{text}

Your evaluation:"""


class CoherenceJudgeTask(ProbingTask):
    """
    LLM self-coherence rating task.

    Each result's `parsed` dict contains:
      - "score"  : int 1–10 (None if parsing failed)
      - "reason" : explanation string
    """

    @property
    def task_name(self) -> str:
        return "coherence_judge"

    def run(
        self,
        corrupted_text: str,
        text_entry: Any,
        granularity: str,
        alpha: float,
        seed: int,
    ) -> list[ProbingResult]:
        user_prompt = _USER_TEMPLATE.format(text=corrupted_text)

        response = self.client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=128,
        )

        score, reason = self._parse_response(response.response_text)

        result = self._make_result(
            response=response,
            text_entry=text_entry,
            granularity=granularity,
            alpha=alpha,
            seed=seed,
            parsed={
                "score": score,
                "reason": reason,
            },
        )
        return [result]

    # ------------------------------------------------------------------
    # Parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> tuple[Optional[int], str]:
        """
        Extract SCORE and REASON from the LLM response.

        Returns (score_int_or_None, reason_string).
        """
        score: Optional[int] = None
        reason: str = ""

        score_match = re.search(r"SCORE\s*:\s*(\d+)", text, re.IGNORECASE)
        if score_match:
            raw = int(score_match.group(1))
            score = max(1, min(10, raw))  # clamp to [1, 10]

        reason_match = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()

        return score, reason

# Made with Bob
