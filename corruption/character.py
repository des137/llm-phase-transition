"""
Character-level text corruptor.

Each character in the text is independently replaced with a random printable
ASCII character with probability (1 - alpha).

Design notes
------------
- Whitespace characters are preserved to keep word boundaries intact and avoid
  creating unreadably fused tokens.  This is a deliberate choice: we want to
  corrupt *content*, not destroy all structure at once.  Set
  `preserve_whitespace=False` to disable this behaviour.
- The replacement pool is printable ASCII (codes 33–126), excluding the
  original character so the replacement is always different.
"""

from __future__ import annotations

import random
import string
from typing import Optional

from .base import BaseCorruptor

# Printable ASCII characters excluding space (we handle whitespace separately)
_PRINTABLE = list(string.printable[:94])  # '!' through '~'


class CharacterCorruptor(BaseCorruptor):
    """
    Character-level corruptor.

    Parameters
    ----------
    preserve_whitespace : bool
        If True (default), whitespace characters (space, newline, tab) are
        never replaced, keeping word boundaries visible.
    """

    def __init__(self, preserve_whitespace: bool = True) -> None:
        self.preserve_whitespace = preserve_whitespace

    # ------------------------------------------------------------------
    # BaseCorruptor interface
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Each character is its own unit."""
        return list(text)

    def _detokenize(self, units: list[str]) -> str:
        return "".join(units)

    def _random_unit(self, original: str, rng: random.Random) -> str:
        if self.preserve_whitespace and original in (" ", "\n", "\t", "\r"):
            return original  # never corrupt whitespace
        # Pick a random printable character that differs from the original
        pool = [c for c in _PRINTABLE if c != original]
        return rng.choice(pool) if pool else original

    # ------------------------------------------------------------------
    # Override corrupt() to honour preserve_whitespace at the unit level
    # ------------------------------------------------------------------

    def corrupt(self, text: str, alpha: float, seed: Optional[int] = None) -> str:
        """
        Corrupt text at the character level.

        Whitespace characters are skipped when preserve_whitespace=True,
        so alpha is applied only to non-whitespace characters.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        rng = random.Random(seed)

        if alpha == 1.0:
            return text

        result: list[str] = []
        for ch in text:
            is_ws = ch in (" ", "\n", "\t", "\r")
            if self.preserve_whitespace and is_ws:
                result.append(ch)
            elif rng.random() < alpha:
                result.append(ch)  # keep original
            else:
                result.append(self._random_unit(ch, rng))
        return "".join(result)

# Made with Bob
