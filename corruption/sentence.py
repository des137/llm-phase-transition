"""
Sentence-level text corruptor.

Each sentence is independently replaced with a noise sentence with probability
(1 - alpha).

Noise sentence strategies (controlled by `noise_mode`):
  "shuffle"   – pick a random sentence from a different position in the same
                text (tests order sensitivity without changing vocabulary)
  "wordsalad" – generate a sentence of random words of similar length drawn
                from the built-in vocabulary (true semantic noise)
  "repeat"    – repeat a random *other* sentence from the text (tests
                redundancy tolerance)

Sentence splitting uses a simple regex heuristic.  For production use,
replace `_tokenize` with an nltk or spacy sentence tokenizer.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from .base import BaseCorruptor
from .word import _DEFAULT_VOCAB

# Regex-based sentence splitter: split on ". ", "! ", "? " followed by capital
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


class SentenceCorruptor(BaseCorruptor):
    """
    Sentence-level corruptor.

    Parameters
    ----------
    noise_mode : str
        "wordsalad" (default) – replace with a random-word sentence
        "shuffle"             – replace with another sentence from the same text
        "repeat"              – repeat a random other sentence from the text
    min_sentence_words : int
        Minimum number of words in a generated noise sentence (default 8).
    max_sentence_words : int
        Maximum number of words in a generated noise sentence (default 20).
    """

    def __init__(
        self,
        noise_mode: str = "wordsalad",
        min_sentence_words: int = 8,
        max_sentence_words: int = 20,
    ) -> None:
        valid_modes = ("wordsalad", "shuffle", "repeat")
        if noise_mode not in valid_modes:
            raise ValueError(f"noise_mode must be one of {valid_modes}, got '{noise_mode}'")
        self.noise_mode = noise_mode
        self.min_sentence_words = min_sentence_words
        self.max_sentence_words = max_sentence_words

        # Will be populated during corrupt() so _random_unit can access them
        self._all_sentences: List[str] = []

    # ------------------------------------------------------------------
    # Override corrupt() to inject sentence list before unit processing
    # ------------------------------------------------------------------

    def corrupt(self, text: str, alpha: float, seed: Optional[int] = None) -> str:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        rng = random.Random(seed)

        if alpha == 1.0:
            return text

        sentences = self._tokenize(text)
        self._all_sentences = sentences  # make available to _random_unit

        corrupted = [
            sent if rng.random() < alpha else self._random_unit(sent, rng)
            for sent in sentences
        ]
        return self._detokenize(corrupted)

    # ------------------------------------------------------------------
    # BaseCorruptor interface
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Split text into sentences using a regex heuristic."""
        sentences = _SENT_SPLIT_RE.split(text.strip())
        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]

    def _detokenize(self, units: list[str]) -> str:
        return " ".join(units)

    def _random_unit(self, original: str, rng: random.Random) -> str:
        if self.noise_mode == "wordsalad":
            return self._word_salad_sentence(original, rng)
        elif self.noise_mode == "shuffle":
            return self._shuffle_sentence(original, rng)
        else:  # "repeat"
            return self._repeat_sentence(original, rng)

    # ------------------------------------------------------------------
    # Noise generation helpers
    # ------------------------------------------------------------------

    def _word_salad_sentence(self, original: str, rng: random.Random) -> str:
        """Generate a sentence of random words with similar length to original."""
        original_word_count = len(original.split())
        low = max(self.min_sentence_words, original_word_count - 3)
        high = max(low, min(self.max_sentence_words, original_word_count + 3))
        target = rng.randint(low, high)
        words = [rng.choice(_DEFAULT_VOCAB) for _ in range(target)]
        # Capitalise first word and add a period
        sentence = " ".join(words).capitalize() + "."
        return sentence

    def _shuffle_sentence(self, original: str, rng: random.Random) -> str:
        """Return a random *other* sentence from the same text."""
        candidates = [s for s in self._all_sentences if s != original]
        if not candidates:
            return self._word_salad_sentence(original, rng)
        return rng.choice(candidates)

    def _repeat_sentence(self, original: str, rng: random.Random) -> str:
        """Repeat a random other sentence (same as shuffle for our purposes)."""
        return self._shuffle_sentence(original, rng)

# Made with Bob
