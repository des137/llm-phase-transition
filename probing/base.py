"""
Abstract base class for all probing tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProbingResult:
    """Holds the raw LLM response and task-specific parsed output."""
    task_name: str
    text_id: str
    granularity: str
    alpha: float
    seed: int
    model: str
    raw_response: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ProbingTask(ABC):
    """
    Base class for a probing task.

    Subclasses implement:
      - `task_name`  : str property
      - `run()`      : send prompt(s) to the LLM and return ProbingResult(s)
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Short identifier for this task, e.g. 'qa', 'summarization'."""

    @abstractmethod
    def run(
        self,
        corrupted_text: str,
        text_entry: Any,          # corpus.loader.TextEntry
        granularity: str,
        alpha: float,
        seed: int,
    ) -> list[ProbingResult]:
        """
        Run the probing task on *corrupted_text* and return a list of
        ProbingResult objects (one per question / sub-task).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_result(
        self,
        response: LLMResponse,
        text_entry: Any,
        granularity: str,
        alpha: float,
        seed: int,
        parsed: Optional[Dict[str, Any]] = None,
    ) -> ProbingResult:
        return ProbingResult(
            task_name=self.task_name,
            text_id=text_entry.id,
            granularity=granularity,
            alpha=alpha,
            seed=seed,
            model=response.model,
            raw_response=response.response_text,
            parsed=parsed or {},
            error=response.error,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds,
        )

# Made with Bob
