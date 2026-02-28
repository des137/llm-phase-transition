"""
Abstract base class for all text corruptors.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional


class BaseCorruptor(ABC):
    """
    Corrupt a text string by replacing a fraction (1 - alpha) of its units
    with random noise.

    Parameters
    ----------
    alpha : float
        Coherence ratio in [0, 1].
        alpha=1.0 → no corruption (original text returned unchanged).
        alpha=0.0 → every unit replaced with noise.
    seed : int | None
        Random seed for reproducibility.
    """

    def corrupt(self, text: str, alpha: float, seed: Optional[int] = None) -> str:
        """
        Return a corrupted version of *text* at coherence ratio *alpha*.

        Parameters
        ----------
        text  : source text string
        alpha : coherence ratio [0, 1]
        seed  : optional RNG seed

        Returns
        -------
        Corrupted text string.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        rng = random.Random(seed)

        if alpha == 1.0:
            return text  # fast path: no corruption

        units = self._tokenize(text)
        corrupted = [
            unit if rng.random() < alpha else self._random_unit(unit, rng)
            for unit in units
        ]
        return self._detokenize(corrupted)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        """Split text into units (characters / words / sentences)."""

    @abstractmethod
    def _detokenize(self, units: list[str]) -> str:
        """Reconstruct text from units."""

    @abstractmethod
    def _random_unit(self, original: str, rng: random.Random) -> str:
        """Return a random replacement for *original*."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def granularity(self) -> str:
        """Human-readable granularity label."""
        return self.__class__.__name__.replace("Corruptor", "").lower()

# Made with Bob
