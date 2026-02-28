"""
Experiment runner — orchestrates the full alpha × granularity × text sweep.

Usage
-----
# Dry run (no API calls, uses built-in sample texts)
python runner.py --dry-run

# Full run with specific models and granularities
python runner.py --models openai:gpt-4o anthropic:claude-3-5-sonnet-20241022 \
                 --granularities char word sentence \
                 --alpha-steps 21 \
                 --seeds 5 \
                 --inference-reps 3

# Quick smoke test (2 texts, 3 alpha levels, 1 seed, 1 rep)
python runner.py --dry-run --max-texts 2 --alpha-steps 3 --seeds 1 --inference-reps 1

Model string format:  "<provider>:<model_id>"
  openai:gpt-4o
  anthropic:claude-3-5-sonnet-20241022
  together:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
  local:meta-llama/Meta-Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from config import (
    ALPHA_LEVELS,
    ALPHA_DENSE_REGION,
    GRANULARITIES,
    MODELS,
    CORRUPTION_SEEDS,
    INFERENCE_REPS,
    INFERENCE_TEMPERATURE,
    RESULTS_DIR,
)
from corpus.loader import CorpusLoader, TextEntry
from corruption import get_corruptor
from llm.client import LLMClient
from probing.qa import QATask
from probing.summarization import SummarizationTask
from probing.coherence_judge import CoherenceJudgeTask
from probing.instruction_follow import InstructionFollowTask
from probing.base import ProbingResult


# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------

def _result_to_dict(r: ProbingResult) -> dict:
    return {
        "task_name": r.task_name,
        "text_id": r.text_id,
        "granularity": r.granularity,
        "alpha": r.alpha,
        "seed": r.seed,
        "inference_rep": r.parsed.get("_inference_rep", 0),
        "model": r.model,
        "raw_response": r.raw_response,
        "parsed": {k: v for k, v in r.parsed.items() if not k.startswith("_")},
        "error": r.error,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "latency_seconds": r.latency_seconds,
    }


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_experiment(
    models: List[str],
    granularities: List[str],
    alpha_levels: List[float],
    texts: List[TextEntry],
    seeds: int,
    inference_reps: int,
    dry_run: bool,
    api_key: Optional[str],
    results_dir: str,
    tasks_to_run: Optional[List[str]] = None,
) -> Dict[str, list]:
    """
    Run the full experiment sweep.

    Parameters
    ----------
    models         : list of "<provider>:<model_id>" strings
    granularities  : list of "char" | "word" | "sentence"
    alpha_levels   : sorted list of coherence ratios [0, 1]
    texts          : list of TextEntry objects
    seeds          : number of independent corruption seeds per condition
    inference_reps : number of LLM inference repetitions per (corrupted_text, task)
                     Rep 0 uses temperature=0 (deterministic baseline).
                     Reps 1..N use INFERENCE_TEMPERATURE for stochastic sampling.
    dry_run        : if True, skip real API calls
    api_key        : optional explicit API key override
    results_dir    : directory to write JSONL result files
    tasks_to_run   : subset of task names; None = all tasks

    Returns
    -------
    dict mapping model_name → list of result dicts
    """
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[str, list] = defaultdict(list)

    total_per_model = (
        len(granularities) * len(alpha_levels) * len(texts) * seeds * inference_reps
    )
    total_conditions = len(models) * total_per_model

    print(
        f"\n{'='*60}\n"
        f"Experiment sweep\n"
        f"  Models          : {models}\n"
        f"  Granularities   : {granularities}\n"
        f"  Alpha levels    : {len(alpha_levels)} ({alpha_levels[0]}–{alpha_levels[-1]})\n"
        f"  Texts           : {len(texts)}\n"
        f"  Corruption seeds: {seeds}\n"
        f"  Inference reps  : {inference_reps}\n"
        f"  Dry run         : {dry_run}\n"
        f"  Total conditions: {total_conditions}\n"
        f"{'='*60}\n"
    )

    for model_name in models:
        print(f"\n[Model: {model_name}]")

        # Rep 0 is deterministic (temperature=0); reps 1..N are stochastic
        clients = _build_clients(
            model_name=model_name,
            api_key=api_key,
            dry_run=dry_run,
            inference_reps=inference_reps,
        )

        output_path = os.path.join(
            results_dir,
            model_name.replace("/", "_").replace(":", "__") + ".jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as out_fh:

            condition_idx = 0
            for granularity in granularities:
                corruptor = get_corruptor(granularity)
                print(f"  Granularity: {granularity}")

                for alpha in alpha_levels:
                    for text_entry in texts:
                        for seed_idx in range(seeds):
                            # Corrupt the text once per (alpha, text, seed)
                            corrupted = corruptor.corrupt(
                                text_entry.text, alpha=alpha, seed=seed_idx
                            )

                            for rep_idx, client in enumerate(clients):
                                condition_idx += 1
                                _print_progress(condition_idx, total_per_model)

                                # Build task instances for this client
                                active_tasks = _build_tasks(client, tasks_to_run)

                                for task in active_tasks:
                                    try:
                                        results = task.run(
                                            corrupted_text=corrupted,
                                            text_entry=text_entry,
                                            granularity=granularity,
                                            alpha=alpha,
                                            seed=seed_idx,
                                        )
                                        for r in results:
                                            # Tag with inference rep index
                                            r.parsed["_inference_rep"] = rep_idx
                                            d = _result_to_dict(r)
                                            out_fh.write(json.dumps(d) + "\n")
                                            all_results[model_name].append(d)
                                    except Exception as exc:  # noqa: BLE001
                                        print(
                                            f"\n    [ERROR] task={task.task_name} "
                                            f"text={text_entry.id} alpha={alpha} "
                                            f"seed={seed_idx} rep={rep_idx}: {exc}"
                                        )

        print(f"\n  Results written to: {output_path}")

    return dict(all_results)


# ---------------------------------------------------------------------------
# Client factory (one per inference rep)
# ---------------------------------------------------------------------------

def _build_clients(
    model_name: str,
    api_key: Optional[str],
    dry_run: bool,
    inference_reps: int,
) -> List[LLMClient]:
    """
    Build a list of LLMClient instances — one per inference repetition.
    Rep 0: temperature=0 (deterministic).
    Reps 1..N: temperature=INFERENCE_TEMPERATURE (stochastic).
    """
    clients = []
    for rep in range(inference_reps):
        temp = 0.0 if rep == 0 else INFERENCE_TEMPERATURE
        clients.append(
            LLMClient(
                model=model_name,
                api_key=api_key,
                dry_run=dry_run,
                temperature=temp,
            )
        )
    return clients


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def _build_tasks(client: LLMClient, tasks_to_run: Optional[List[str]]):
    all_tasks = {
        "qa": QATask(client),
        "summarization": SummarizationTask(client),
        "coherence_judge": CoherenceJudgeTask(client),
        "instruction_follow": InstructionFollowTask(client),
    }
    if tasks_to_run is None:
        return list(all_tasks.values())
    return [all_tasks[t] for t in tasks_to_run if t in all_tasks]


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _print_progress(current: int, total: int) -> None:
    pct = 100 * current / total if total else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r    [{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM comprehension phase transition experiment."
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help='Model strings, e.g. "openai:gpt-4o" "anthropic:claude-3-5-sonnet-20241022"'
    )
    parser.add_argument(
        "--granularities", nargs="+", default=GRANULARITIES,
        choices=["char", "word", "sentence"],
        help="Granularity levels to sweep"
    )
    parser.add_argument(
        "--alpha-steps", type=int, default=21,
        help="Number of alpha levels (evenly spaced 0→1, default 21)"
    )
    parser.add_argument(
        "--dense-alpha", action="store_true",
        help="Also include the dense alpha grid around the critical region"
    )
    parser.add_argument(
        "--seeds", type=int, default=CORRUPTION_SEEDS,
        help="Number of corruption seeds per condition"
    )
    parser.add_argument(
        "--inference-reps", type=int, default=INFERENCE_REPS,
        help=(
            "Number of LLM inference repetitions per (corrupted_text, task). "
            "Rep 0 is deterministic (temp=0); reps 1..N use INFERENCE_TEMPERATURE. "
            "Use ≥3 to get meaningful dispersion estimates."
        )
    )
    parser.add_argument(
        "--max-texts", type=int, default=None,
        help="Cap the number of texts (useful for quick tests)"
    )
    parser.add_argument(
        "--corpus-json", type=str, default=None,
        help="Path to a JSON corpus file (default: built-in sample texts)"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["qa", "summarization", "coherence_judge", "instruction_follow"],
        default=None,
        help="Subset of tasks to run (default: all)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR,
        help="Directory to write result JSONL files"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key override (provider-specific; usually set via .env)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip real LLM calls; use mock responses"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build alpha levels
    import numpy as np  # type: ignore[import]
    alpha_levels = [round(v, 2) for v in np.linspace(0.0, 1.0, args.alpha_steps).tolist()]
    if args.dense_alpha:
        alpha_levels = sorted(set(alpha_levels + ALPHA_DENSE_REGION))

    # Load corpus
    loader = CorpusLoader(
        json_path=args.corpus_json,
        max_texts=args.max_texts,
    )
    texts = loader.load()
    if not texts:
        print("ERROR: No texts loaded. Check your corpus source.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(texts)} texts from corpus.")

    t0 = time.perf_counter()
    run_experiment(
        models=args.models,
        granularities=args.granularities,
        alpha_levels=alpha_levels,
        texts=texts,
        seeds=args.seeds,
        inference_reps=args.inference_reps,
        dry_run=args.dry_run,
        api_key=args.api_key,
        results_dir=args.results_dir,
        tasks_to_run=args.tasks,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nExperiment complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()

# Made with Bob
