"""
Central configuration for the LLM Comprehension Phase Transition experiment.
All tunable parameters live here so runner.py and analysis.py stay clean.

Environment variables (loaded from .env if present) override defaults.
Copy .env.example → .env and fill in your API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# Load .env file if present (silently ignored if python-dotenv not installed)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # dotenv not installed; rely on shell environment variables

import numpy as np


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------

ALPHA_LEVELS: List[float] = [round(v, 2) for v in np.linspace(0.0, 1.0, 21).tolist()]
"""21 coherence-ratio levels: 0.00, 0.05, 0.10, ..., 1.00"""

ALPHA_DENSE_REGION: List[float] = [round(v, 2) for v in np.linspace(0.40, 0.70, 13).tolist()]
"""Denser grid around the suspected critical region for fine-grained analysis."""


# ---------------------------------------------------------------------------
# Granularity levels
# ---------------------------------------------------------------------------

GRANULARITIES = ["char", "word", "sentence"]


# ---------------------------------------------------------------------------
# Models — provider:model_id format
# ---------------------------------------------------------------------------
# Provider prefixes:
#   openai:   OpenAI API          (requires OPENAI_API_KEY)
#   anthropic: Anthropic API      (requires ANTHROPIC_API_KEY)
#   together: Together AI API     (requires TOGETHER_API_KEY)
#   local:    Local vllm/Ollama   (requires LOCAL_LLM_BASE_URL)
#
# If no prefix is given, "openai:" is assumed for backward compatibility.

MODELS = [
    "openai:gpt-4o",
    "openai:gpt-3.5-turbo",
    # "anthropic:claude-3-5-sonnet-20241022",
    # "together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    # "together:mistralai/Mistral-7B-Instruct-v0.3",
    # "local:meta-llama/Meta-Llama-3-8B-Instruct",
]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

CORPUS_DOMAINS = ["encyclopedic", "narrative", "scientific", "conversational", "procedural"]
TEXTS_PER_DOMAIN = 20          # target; sample texts module provides fewer for unit tests
MIN_TEXT_WORDS = 200
MAX_TEXT_WORDS = 600


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

CORRUPTION_SEEDS = int(os.environ.get("CORRUPTION_SEEDS", "5"))
"""Independent corruption samples per (α, granularity, text)."""

RANDOM_WORD_VOCAB_SIZE = 50_000

# ---------------------------------------------------------------------------
# Inference repetitions (for dispersion measurement)
# ---------------------------------------------------------------------------

INFERENCE_REPS = int(os.environ.get("INFERENCE_REPS", "3"))
"""
Number of times each LLM call is repeated for the *same* corrupted text.
Repetitions use temperature > 0 so responses vary, giving us a distribution
of scores per condition from which we can compute mean ± std / CI.

Set to 1 to disable repetitions (deterministic mode, temperature=0).
"""

INFERENCE_TEMPERATURE = float(os.environ.get("INFERENCE_TEMPERATURE", "0.7"))
"""
Sampling temperature for inference repetitions.
Only used when INFERENCE_REPS > 1.  When INFERENCE_REPS == 1, temperature=0.
"""


# ---------------------------------------------------------------------------
# Probing tasks
# ---------------------------------------------------------------------------

QA_QUESTIONS_PER_TEXT = 5      # pre-generated factual questions per source text
INSTRUCTION_KEYWORD = "BANANA" # keyword the LLM must echo in Task D


# ---------------------------------------------------------------------------
# LLM call settings
# ---------------------------------------------------------------------------

MAX_TOKENS_QA = 128
MAX_TOKENS_SUMMARY = 256
MAX_TOKENS_COHERENCE = 128
MAX_TOKENS_INSTRUCTION = 64
TEMPERATURE = 0.0              # deterministic for reproducibility
REQUEST_TIMEOUT = 60           # seconds


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")


# ---------------------------------------------------------------------------
# Dataclass for a single experiment condition
# ---------------------------------------------------------------------------

@dataclass
class ExperimentCondition:
    text_id: str
    domain: str
    granularity: str          # "char" | "word" | "sentence"
    alpha: float              # coherence ratio [0, 1]
    seed: int                 # corruption seed index


@dataclass
class ExperimentConfig:
    models: List[str] = field(default_factory=lambda: MODELS)
    granularities: List[str] = field(default_factory=lambda: GRANULARITIES)
    alpha_levels: List[float] = field(default_factory=lambda: ALPHA_LEVELS)
    corruption_seeds: int = CORRUPTION_SEEDS
    inference_reps: int = INFERENCE_REPS
    dry_run: bool = False      # if True, skip actual LLM calls; return mock responses
    dense_alpha: bool = False  # if True, also sweep ALPHA_DENSE_REGION
    max_texts: int | None = None  # cap number of texts (useful for quick tests)

# Made with Bob
