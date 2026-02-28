"""
Unit tests for the text corruption modules.

Run with:
    cd paper_idea/experiment
    pytest tests/test_corruption.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from corruption.character import CharacterCorruptor
from corruption.word import WordCorruptor
from corruption.sentence import SentenceCorruptor
from corruption import get_corruptor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a bright cold day in April. "
    "The French Revolution began in 1789. "
    "Napoleon Bonaparte rose to prominence during this period."
)


# ===========================================================================
# CharacterCorruptor
# ===========================================================================

class TestCharacterCorruptor:

    def test_alpha_one_returns_original(self):
        c = CharacterCorruptor()
        result = c.corrupt(SAMPLE_TEXT, alpha=1.0, seed=0)
        assert result == SAMPLE_TEXT

    def test_alpha_zero_changes_all_non_whitespace(self):
        c = CharacterCorruptor(preserve_whitespace=True)
        result = c.corrupt(SAMPLE_TEXT, alpha=0.0, seed=42)
        # Whitespace positions must be preserved
        for orig_ch, res_ch in zip(SAMPLE_TEXT, result):
            if orig_ch in (" ", "\n", "\t"):
                assert res_ch == orig_ch, "Whitespace should be preserved"
        # At least some characters should differ
        non_ws_orig = [c for c in SAMPLE_TEXT if c not in " \n\t"]
        non_ws_res = [c for c in result if c not in " \n\t"]
        assert non_ws_orig != non_ws_res, "Non-whitespace chars should change at alpha=0"

    def test_length_preserved(self):
        c = CharacterCorruptor()
        for alpha in [0.0, 0.5, 1.0]:
            result = c.corrupt(SAMPLE_TEXT, alpha=alpha, seed=0)
            assert len(result) == len(SAMPLE_TEXT), f"Length changed at alpha={alpha}"

    def test_deterministic_with_same_seed(self):
        c = CharacterCorruptor()
        r1 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=7)
        r2 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=7)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        c = CharacterCorruptor()
        r1 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=1)
        r2 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=2)
        assert r1 != r2

    def test_intermediate_alpha_partial_corruption(self):
        c = CharacterCorruptor()
        result = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=0)
        # Should differ from original but not be completely different
        assert result != SAMPLE_TEXT
        matching = sum(a == b for a, b in zip(SAMPLE_TEXT, result))
        total = len(SAMPLE_TEXT)
        # Roughly 50% should match (allow wide tolerance for randomness)
        assert 0.2 * total < matching < 0.9 * total

    def test_invalid_alpha_raises(self):
        c = CharacterCorruptor()
        with pytest.raises(ValueError):
            c.corrupt(SAMPLE_TEXT, alpha=1.5)
        with pytest.raises(ValueError):
            c.corrupt(SAMPLE_TEXT, alpha=-0.1)

    def test_granularity_label(self):
        assert CharacterCorruptor().granularity == "character"


# ===========================================================================
# WordCorruptor
# ===========================================================================

class TestWordCorruptor:

    def test_alpha_one_returns_original(self):
        c = WordCorruptor()
        result = c.corrupt(SAMPLE_TEXT, alpha=1.0, seed=0)
        assert result == SAMPLE_TEXT

    def test_alpha_zero_changes_all_words(self):
        c = WordCorruptor()
        result = c.corrupt(SAMPLE_TEXT, alpha=0.0, seed=0)
        orig_words = SAMPLE_TEXT.split()
        res_words = result.split()
        assert len(orig_words) == len(res_words), "Word count should be preserved"
        # All words should differ (with overwhelming probability)
        matches = sum(o == r for o, r in zip(orig_words, res_words))
        assert matches < len(orig_words) * 0.2, "Most words should change at alpha=0"

    def test_word_count_preserved(self):
        c = WordCorruptor()
        for alpha in [0.0, 0.5, 1.0]:
            result = c.corrupt(SAMPLE_TEXT, alpha=alpha, seed=0)
            assert len(result.split()) == len(SAMPLE_TEXT.split())

    def test_deterministic_with_same_seed(self):
        c = WordCorruptor()
        r1 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=3)
        r2 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=3)
        assert r1 == r2

    def test_random_mode(self):
        c = WordCorruptor(noise_mode="random")
        result = c.corrupt(SAMPLE_TEXT, alpha=0.0, seed=0)
        assert result != SAMPLE_TEXT

    def test_invalid_noise_mode(self):
        with pytest.raises(ValueError):
            WordCorruptor(noise_mode="invalid")

    def test_granularity_label(self):
        assert WordCorruptor().granularity == "word"

    def test_split_punctuation(self):
        leading, core, trailing = WordCorruptor._split_punctuation('"hello,')
        assert leading == '"'
        assert core == "hello"
        assert trailing == ","

    def test_split_punctuation_plain_word(self):
        leading, core, trailing = WordCorruptor._split_punctuation("hello")
        assert leading == ""
        assert core == "hello"
        assert trailing == ""


# ===========================================================================
# SentenceCorruptor
# ===========================================================================

class TestSentenceCorruptor:

    def test_alpha_one_returns_original(self):
        c = SentenceCorruptor()
        result = c.corrupt(SAMPLE_TEXT, alpha=1.0, seed=0)
        assert result == SAMPLE_TEXT

    def test_alpha_zero_changes_sentences(self):
        c = SentenceCorruptor(noise_mode="wordsalad")
        result = c.corrupt(SAMPLE_TEXT, alpha=0.0, seed=0)
        assert result != SAMPLE_TEXT

    def test_deterministic_with_same_seed(self):
        c = SentenceCorruptor()
        r1 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=5)
        r2 = c.corrupt(SAMPLE_TEXT, alpha=0.5, seed=5)
        assert r1 == r2

    def test_shuffle_mode(self):
        c = SentenceCorruptor(noise_mode="shuffle")
        result = c.corrupt(SAMPLE_TEXT, alpha=0.0, seed=0)
        # All sentences should come from the original text
        orig_sentences = set(c._tokenize(SAMPLE_TEXT))
        result_sentences = set(c._tokenize(result))
        assert result_sentences.issubset(orig_sentences)

    def test_invalid_noise_mode(self):
        with pytest.raises(ValueError):
            SentenceCorruptor(noise_mode="invalid")

    def test_granularity_label(self):
        assert SentenceCorruptor().granularity == "sentence"

    def test_tokenize_splits_on_sentence_boundaries(self):
        c = SentenceCorruptor()
        sentences = c._tokenize(SAMPLE_TEXT)
        assert len(sentences) >= 2, "Should split into multiple sentences"
        for s in sentences:
            assert s.strip() != "", "No empty sentences"


# ===========================================================================
# Factory
# ===========================================================================

class TestGetCorruptor:

    def test_char(self):
        c = get_corruptor("char")
        assert isinstance(c, CharacterCorruptor)

    def test_word(self):
        c = get_corruptor("word")
        assert isinstance(c, WordCorruptor)

    def test_sentence(self):
        c = get_corruptor("sentence")
        assert isinstance(c, SentenceCorruptor)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_corruptor("paragraph")


# ===========================================================================
# Cross-corruptor: monotonicity of corruption with alpha
# ===========================================================================

class TestMonotonicity:
    """
    As alpha decreases, the text should become more different from the original.
    We measure this as the fraction of matching tokens.
    """

    @pytest.mark.parametrize("corruptor_cls,split_fn", [
        (CharacterCorruptor, list),
        (WordCorruptor, str.split),
    ])
    def test_more_corruption_at_lower_alpha(self, corruptor_cls, split_fn):
        c = corruptor_cls()
        alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
        match_rates = []
        for alpha in alphas:
            corrupted = c.corrupt(SAMPLE_TEXT, alpha=alpha, seed=0)
            orig_units = split_fn(SAMPLE_TEXT)
            corr_units = split_fn(corrupted)
            n = min(len(orig_units), len(corr_units))
            matches = sum(a == b for a, b in zip(orig_units[:n], corr_units[:n]))
            match_rates.append(matches / n if n else 0.0)

        # Match rate should be non-increasing as alpha decreases
        for i in range(len(match_rates) - 1):
            assert match_rates[i] >= match_rates[i + 1] - 0.05, (
                f"Match rate not monotone: alpha={alphas[i]}→{alphas[i+1]}, "
                f"rates={match_rates[i]:.3f}→{match_rates[i+1]:.3f}"
            )

# Made with Bob
