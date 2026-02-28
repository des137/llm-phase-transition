"""
Task A — Closed-Form Question Answering.

For each QA pair attached to the source text, ask the LLM to answer the
question given the (possibly corrupted) text as context.  Returns one
ProbingResult per question.
"""

from __future__ import annotations

from typing import Any

from .base import ProbingTask, ProbingResult
from llm.client import LLMClient

_SYSTEM_PROMPT = (
    "You are a reading comprehension assistant. "
    "Answer the question using ONLY information from the provided passage. "
    "Give a short, direct answer — a few words or a single sentence. "
    "If the passage does not contain enough information to answer, reply exactly: CANNOT ANSWER"
)

_USER_TEMPLATE = """\
PASSAGE:
{text}

QUESTION: {question}

Answer:"""


class QATask(ProbingTask):
    """
    Closed-form QA probing task.

    Returns one ProbingResult per (question, corrupted_text) pair.
    Each result's `parsed` dict contains:
      - "question"       : the question string
      - "expected_answer": ground-truth answer
      - "predicted"      : LLM's answer (stripped)
      - "refused"        : True if LLM said CANNOT ANSWER
    """

    @property
    def task_name(self) -> str:
        return "qa"

    def run(
        self,
        corrupted_text: str,
        text_entry: Any,
        granularity: str,
        alpha: float,
        seed: int,
    ) -> list[ProbingResult]:
        results: list[ProbingResult] = []

        if not text_entry.questions:
            # No QA pairs available for this text — return empty
            return results

        for qa in text_entry.questions:
            user_prompt = _USER_TEMPLATE.format(
                text=corrupted_text,
                question=qa.question,
            )
            response = self.client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=128,
            )

            predicted = response.response_text.strip() if response.ok else ""
            refused = predicted.upper().startswith("CANNOT ANSWER")

            result = self._make_result(
                response=response,
                text_entry=text_entry,
                granularity=granularity,
                alpha=alpha,
                seed=seed,
                parsed={
                    "question": qa.question,
                    "expected_answer": qa.answer,
                    "predicted": predicted,
                    "refused": refused,
                },
            )
            results.append(result)

        return results

# Made with Bob
