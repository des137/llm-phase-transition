"""
Corpus loader: loads and validates source texts for the experiment.

Supports:
  - Built-in sample texts (for dry runs / unit tests)
  - JSON file with the same schema as SAMPLE_TEXTS
  - Plain directory of .txt files (no QA pairs; QA generation is out of scope here)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .sample_texts import SAMPLE_TEXTS


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class TextEntry:
    id: str
    domain: str
    text: str
    questions: List[QAPair] = field(default_factory=list)
    reference_summary: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextEntry":
        return cls(
            id=d["id"],
            domain=d["domain"],
            text=d["text"],
            questions=[QAPair(**q) for q in d.get("questions", [])],
            reference_summary=d.get("reference_summary", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "text": self.text,
            "questions": [{"question": q.question, "answer": q.answer} for q in self.questions],
            "reference_summary": self.reference_summary,
        }

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class CorpusLoader:
    """
    Load a corpus of TextEntry objects from various sources.

    Usage
    -----
    # Built-in sample texts
    loader = CorpusLoader()
    texts = loader.load()

    # From a JSON file
    loader = CorpusLoader(json_path="my_corpus.json")
    texts = loader.load()

    # From a directory of .txt files (no QA pairs)
    loader = CorpusLoader(txt_dir="corpus/")
    texts = loader.load()
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        txt_dir: Optional[str] = None,
        max_texts: Optional[int] = None,
        min_words: int = 50,
        max_words: int = 2000,
    ) -> None:
        self.json_path = json_path
        self.txt_dir = txt_dir
        self.max_texts = max_texts
        self.min_words = min_words
        self.max_words = max_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[TextEntry]:
        """Return a list of TextEntry objects according to the configured source."""
        if self.json_path:
            entries = self._load_from_json(self.json_path)
        elif self.txt_dir:
            entries = self._load_from_txt_dir(self.txt_dir)
        else:
            entries = self._load_sample_texts()

        entries = self._filter(entries)

        if self.max_texts is not None:
            entries = entries[: self.max_texts]

        return entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_sample_texts(self) -> List[TextEntry]:
        return [TextEntry.from_dict(d) for d in SAMPLE_TEXTS]

    def _load_from_json(self, path: str) -> List[TextEntry]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Corpus JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Corpus JSON must be a list of text entry objects.")
        return [TextEntry.from_dict(d) for d in data]

    def _load_from_txt_dir(self, directory: str) -> List[TextEntry]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Corpus directory not found: {directory}")
        entries: List[TextEntry] = []
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(directory, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            text_id = os.path.splitext(fname)[0]
            # Infer domain from filename prefix (e.g. "enc_001.txt" → "encyclopedic")
            domain = self._infer_domain(text_id)
            entries.append(TextEntry(id=text_id, domain=domain, text=text))
        return entries

    def _filter(self, entries: List[TextEntry]) -> List[TextEntry]:
        filtered = []
        for e in entries:
            wc = e.word_count
            if wc < self.min_words:
                continue
            if wc > self.max_words:
                # Truncate to max_words
                e.text = " ".join(e.text.split()[: self.max_words])
            filtered.append(e)
        return filtered

    @staticmethod
    def _infer_domain(text_id: str) -> str:
        prefix_map = {
            "enc": "encyclopedic",
            "nar": "narrative",
            "sci": "scientific",
            "conv": "conversational",
            "proc": "procedural",
        }
        prefix = text_id.split("_")[0] if "_" in text_id else ""
        return prefix_map.get(prefix, "unknown")

# Made with Bob
