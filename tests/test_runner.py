"""
Integration / smoke tests for the experiment runner.

These tests use dry_run=True so no real LLM API calls are made.
They verify that the full pipeline (corpus → corruption → probing → results)
runs without errors and produces the expected output structure.

Run with:
    cd paper_idea/experiment
    pytest tests/test_runner.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import pytest

from corpus.loader import CorpusLoader
from runner import run_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_texts():
    """Load the built-in sample texts (no external files needed)."""
    loader = CorpusLoader(max_texts=2)
    return loader.load()


@pytest.fixture
def temp_results_dir():
    """Provide a temporary directory for result files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestRunnerSmoke:

    def test_dry_run_completes(self, sample_texts, temp_results_dir):
        """Full pipeline should complete without errors in dry-run mode."""
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=["word"],
            alpha_levels=[0.0, 0.5, 1.0],
            texts=sample_texts,
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["qa", "instruction_follow"],
        )
        assert "openai:gpt-4o" in results
        assert len(results["openai:gpt-4o"]) > 0

    def test_results_have_required_fields(self, sample_texts, temp_results_dir):
        """Every result record must have the required fields."""
        required_fields = {
            "task_name", "text_id", "granularity", "alpha",
            "seed", "model", "raw_response", "parsed",
        }
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=["char"],
            alpha_levels=[1.0],
            texts=sample_texts[:1],
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["qa"],
        )
        for record in results["openai:gpt-4o"]:
            for field in required_fields:
                assert field in record, f"Missing field '{field}' in result"

    def test_jsonl_file_written(self, sample_texts, temp_results_dir):
        """A JSONL file should be written for each model."""
        run_experiment(
            models=["openai:gpt-3.5-turbo"],
            granularities=["sentence"],
            alpha_levels=[0.5, 1.0],
            texts=sample_texts[:1],
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["summarization"],
        )
        expected_file = os.path.join(temp_results_dir, "openai__gpt-3.5-turbo.jsonl")
        assert os.path.isfile(expected_file), "JSONL file not created"

        # Verify each line is valid JSON
        with open(expected_file, "r") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        assert len(lines) > 0
        for line in lines:
            record = json.loads(line)  # should not raise
            assert "task_name" in record

    def test_alpha_values_in_results(self, sample_texts, temp_results_dir):
        """Results should contain all requested alpha values."""
        alpha_levels = [0.0, 0.5, 1.0]
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=["word"],
            alpha_levels=alpha_levels,
            texts=sample_texts[:1],
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["coherence_judge"],
        )
        found_alphas = {r["alpha"] for r in results["openai:gpt-4o"]}
        for alpha in alpha_levels:
            assert alpha in found_alphas, f"Alpha {alpha} not found in results"

    def test_multiple_granularities(self, sample_texts, temp_results_dir):
        """Results should contain all requested granularities."""
        granularities = ["char", "word", "sentence"]
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=granularities,
            alpha_levels=[1.0],
            texts=sample_texts[:1],
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["qa"],
        )
        found_granularities = {r["granularity"] for r in results["openai:gpt-4o"]}
        for g in granularities:
            assert g in found_granularities, f"Granularity '{g}' not found in results"

    def test_multiple_seeds_produce_different_corruptions(self, sample_texts, temp_results_dir):
        """
        With seeds > 1, the raw_response should vary across seeds
        (because the corrupted text differs, so the mock hash differs).
        """
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=["word"],
            alpha_levels=[0.5],
            texts=sample_texts[:1],
            seeds=3,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["qa"],
        )
        responses = [r["raw_response"] for r in results["openai:gpt-4o"]]
        # With 3 seeds and 5 QA questions, we expect at least some variation
        assert len(set(responses)) > 1, "All responses identical across seeds — unexpected"

    def test_dry_run_response_format(self, sample_texts, temp_results_dir):
        """Dry-run responses should contain the DRY-RUN marker."""
        results = run_experiment(
            models=["openai:gpt-4o"],
            granularities=["word"],
            alpha_levels=[1.0],
            texts=sample_texts[:1],
            seeds=1,
            inference_reps=1,
            dry_run=True,
            api_key=None,
            results_dir=temp_results_dir,
            tasks_to_run=["summarization"],
        )
        for record in results["openai:gpt-4o"]:
            assert "DRY-RUN" in record["raw_response"], (
                "Dry-run response should contain DRY-RUN marker"
            )


# ---------------------------------------------------------------------------
# Corpus loader tests
# ---------------------------------------------------------------------------

class TestCorpusLoader:

    def test_loads_sample_texts(self):
        loader = CorpusLoader()
        texts = loader.load()
        assert len(texts) == 5  # 5 built-in sample texts

    def test_max_texts_cap(self):
        loader = CorpusLoader(max_texts=2)
        texts = loader.load()
        assert len(texts) == 2

    def test_text_entry_fields(self):
        loader = CorpusLoader()
        texts = loader.load()
        for t in texts:
            assert t.id
            assert t.domain
            assert t.text
            assert len(t.questions) > 0
            assert t.reference_summary

    def test_word_count_property(self):
        loader = CorpusLoader()
        texts = loader.load()
        for t in texts:
            assert t.word_count == len(t.text.split())

    def test_min_words_filter(self):
        loader = CorpusLoader(min_words=10_000)  # impossibly high
        texts = loader.load()
        assert len(texts) == 0

    def test_json_not_found_raises(self):
        loader = CorpusLoader(json_path="/nonexistent/path.json")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_txt_dir_not_found_raises(self):
        loader = CorpusLoader(txt_dir="/nonexistent/dir/")
        with pytest.raises(NotADirectoryError):
            loader.load()

# Made with Bob
