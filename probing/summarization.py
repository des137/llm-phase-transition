"""
Task B — Summarization.

Ask the LLM to summarize the (possibly corrupted) text in 2–3 sentences.
Returns a single ProbingResult whose `parsed` dict contains the generated
summary alongside the reference summary for downstream metric computation.
"""

from __future__ import annotations

from typing import Any

from .base import ProbingTask, ProbingResult

_SYSTEM_PROMPT = (
    "You are a concise summarization assistant. "
    "Summarize the provided passage in 2–3 sentences, capturing the main idea. "
    "Write only the summary — no preamble, no commentary."
)

_USER_TEMPLATE = """\
PASSAGE:
{text}

Summary:"""


class SummarizationTask(ProbingTask):
    """
    Summarization probing task.

    Each result's `parsed` dict contains:
      - "summary"           : LLM-generated summary
      - "reference_summary" : ground-truth reference summary
      - "refused"           : True if the LLM declined to summarize
    """

    @property
    def task_name(self) -> str:
        return "summarization"

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
            max_tokens=256,
        )

        summary = response.response_text.strip() if response.ok else ""
        refused = (
            not summary
            or any(
                phrase in summary.lower()
                for phrase in (
                    "cannot summarize",
                    "unable to summarize",
                    "does not make sense",
                    "no coherent",
                    "incomprehensible",
                )
            )
        )

        result = self._make_result(
            response=response,
            text_entry=text_entry,
            granularity=granularity,
            alpha=alpha,
            seed=seed,
            parsed={
                "summary": summary,
                "reference_summary": text_entry.reference_summary,
                "refused": refused,
            },
        )
        return [result]

# Made with Bob
