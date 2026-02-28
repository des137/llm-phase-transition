"""
Word-level text corruptor.

Each whitespace-delimited token is independently replaced with a random word
drawn from a built-in vocabulary with probability (1 - alpha).

Design notes
------------
- Punctuation attached to a word (e.g. "hello,") is preserved on the
  replacement token so sentence structure remains superficially intact.
- The built-in vocabulary is a ~1 000-word sample of common English words.
  For the full experiment, swap in a 50 k-word list via `vocab` parameter.
- `noise_mode` controls the replacement strategy:
    "vocab"   – draw from the vocabulary list (default)
    "random"  – generate a random lowercase string of similar length
"""

from __future__ import annotations

import random
import re
import string
from typing import List, Optional

from .base import BaseCorruptor

# ---------------------------------------------------------------------------
# Built-in vocabulary (common English words, ~300 entries for portability)
# Extend this list or pass a custom vocab to WordCorruptor.__init__
# ---------------------------------------------------------------------------
_DEFAULT_VOCAB: List[str] = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "great", "between", "need",
    "large", "often", "hand", "high", "place", "hold", "turn", "help",
    "part", "small", "number", "off", "always", "move", "live", "point",
    "city", "play", "small", "number", "off", "always", "move", "live",
    "world", "still", "nation", "tell", "man", "found", "here", "those",
    "too", "went", "old", "long", "down", "never", "each", "much", "before",
    "right", "too", "mean", "same", "differ", "tell", "does", "set", "three",
    "air", "put", "end", "why", "again", "turn", "here", "off", "went",
    "read", "need", "land", "different", "home", "move", "try", "kind",
    "hand", "picture", "change", "play", "spell", "air", "away", "animal",
    "house", "point", "page", "letter", "mother", "answer", "found", "study",
    "still", "learn", "plant", "cover", "food", "sun", "four", "between",
    "state", "keep", "eye", "never", "last", "let", "thought", "city",
    "tree", "cross", "farm", "hard", "start", "might", "story", "saw",
    "far", "sea", "draw", "left", "late", "run", "while", "press", "close",
    "night", "real", "life", "few", "north", "open", "seem", "together",
    "next", "white", "children", "begin", "got", "walk", "example", "ease",
    "paper", "group", "always", "music", "those", "both", "mark", "book",
    "carry", "took", "science", "eat", "room", "friend", "began", "idea",
    "fish", "mountain", "stop", "once", "base", "hear", "horse", "cut",
    "sure", "watch", "color", "face", "wood", "main", "enough", "plain",
    "girl", "usual", "young", "ready", "above", "ever", "red", "list",
    "though", "feel", "talk", "bird", "soon", "body", "dog", "family",
    "direct", "pose", "leave", "song", "measure", "door", "product", "black",
    "short", "numeral", "class", "wind", "question", "happen", "complete",
    "ship", "area", "half", "rock", "order", "fire", "south", "problem",
    "piece", "told", "knew", "pass", "since", "top", "whole", "king",
    "space", "heard", "best", "hour", "better", "true", "during", "hundred",
    "five", "remember", "step", "early", "hold", "west", "ground", "interest",
    "reach", "fast", "verb", "sing", "listen", "six", "table", "travel",
    "less", "morning", "ten", "simple", "several", "vowel", "toward", "war",
    "lay", "against", "pattern", "slow", "center", "love", "person", "money",
    "serve", "appear", "road", "map", "rain", "rule", "govern", "pull",
    "cold", "notice", "voice", "unit", "power", "town", "fine", "drive",
    "lead", "cry", "dark", "machine", "note", "wait", "plan", "figure",
    "star", "box", "noun", "field", "rest", "correct", "able", "pound",
    "done", "beauty", "drive", "stood", "contain", "front", "teach", "week",
    "final", "gave", "green", "oh", "quick", "develop", "ocean", "warm",
    "free", "minute", "strong", "special", "mind", "behind", "clear",
    "tail", "produce", "fact", "street", "inch", "multiply", "nothing",
    "course", "stay", "wheel", "full", "force", "blue", "object", "decide",
    "surface", "deep", "moon", "island", "foot", "system", "busy", "test",
    "record", "boat", "common", "gold", "possible", "plane", "stead",
    "dry", "wonder", "laugh", "thousand", "ago", "ran", "check", "game",
    "shape", "equate", "hot", "miss", "brought", "heat", "snow", "tire",
    "bring", "yes", "distant", "fill", "east", "paint", "language", "among",
]


class WordCorruptor(BaseCorruptor):
    """
    Word-level corruptor.

    Parameters
    ----------
    vocab : list[str] | None
        Vocabulary to draw random replacement words from.
        Defaults to the built-in ~300-word list.
    noise_mode : str
        "vocab"  – replace with a word from *vocab* (default)
        "random" – replace with a random lowercase string of similar length
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        noise_mode: str = "vocab",
    ) -> None:
        if noise_mode not in ("vocab", "random"):
            raise ValueError(f"noise_mode must be 'vocab' or 'random', got '{noise_mode}'")
        self.vocab = vocab if vocab is not None else _DEFAULT_VOCAB
        self.noise_mode = noise_mode

    # ------------------------------------------------------------------
    # BaseCorruptor interface
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Split on whitespace, preserving tokens (including punctuation)."""
        return text.split()

    def _detokenize(self, units: list[str]) -> str:
        return " ".join(units)

    def _random_unit(self, original: str, rng: random.Random) -> str:
        # Detect leading/trailing punctuation to preserve superficial structure
        leading, core, trailing = self._split_punctuation(original)
        target_len = max(1, len(core))

        if self.noise_mode == "vocab":
            replacement = rng.choice(self.vocab)
        else:
            # Random lowercase string of similar length
            length = rng.randint(max(1, target_len - 2), target_len + 2)
            replacement = "".join(rng.choices(string.ascii_lowercase, k=length))

        return leading + replacement + trailing

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_punctuation(token: str):
        """
        Split a token into (leading_punct, core, trailing_punct).
        E.g. '"hello,' → ('"', 'hello', ',')
        """
        leading = ""
        trailing = ""
        core = token

        # Strip leading punctuation
        i = 0
        while i < len(core) and core[i] in string.punctuation:
            leading += core[i]
            i += 1
        core = core[i:]

        # Strip trailing punctuation
        j = len(core) - 1
        while j >= 0 and core[j] in string.punctuation:
            trailing = core[j] + trailing
            j -= 1
        core = core[: j + 1]

        return leading, core, trailing

# Made with Bob
