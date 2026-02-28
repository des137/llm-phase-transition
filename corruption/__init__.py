from .character import CharacterCorruptor
from .word import WordCorruptor
from .sentence import SentenceCorruptor
from .base import BaseCorruptor

__all__ = ["BaseCorruptor", "CharacterCorruptor", "WordCorruptor", "SentenceCorruptor"]


def get_corruptor(granularity: str) -> "BaseCorruptor":
    """Factory: return the corruptor for the given granularity string."""
    mapping = {
        "char": CharacterCorruptor,
        "word": WordCorruptor,
        "sentence": SentenceCorruptor,
    }
    if granularity not in mapping:
        raise ValueError(f"Unknown granularity '{granularity}'. Choose from: {list(mapping)}")
    return mapping[granularity]()

# Made with Bob
