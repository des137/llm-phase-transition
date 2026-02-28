"""
Concurrent experiment runner — same logic as runner.py but uses a
ThreadPoolExecutor to fire multiple LLM calls in parallel.

Usage
-----
# Full run, 20 concurrent workers
python runner_async.py --models openai:gpt-4o \
                       --granularities char word sentence \
                       --alpha-steps 11 --seeds 3 --inference-reps 2 \
                       --concurrency 20

# Quick smoke test
python runner_async.py --dry-run --max-texts 2 --alpha-steps 3 --seeds 1 \
                       --inference-reps 1 --concurrency 4

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
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
# Work unit
# ---------------------------------------------------------------------------

@dataclass
class _WorkUnit:
    """One (granularity, alpha, text, seed, rep_idx) condition."""
    granularity: str
    alpha: float
    text_entry: TextEntry
    seed_idx: int
    rep_idx: int
    corrupted: str
    client: LLMClient
    tasks_to_run: Optional[List[str]]


# ---------------------------------------------------------------------------
# Result serialisation
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
# Worker function (runs in a thread)
# ---------------------------------------------------------------------------

def _run_work_unit(unit: _WorkUnit) -> List[dict]:
    """Execute all tasks for one work unit. Returns list of result dicts."""
    all_task_classes = {
        "qa": QATask,
        "summarization": SummarizationTask,
        "coherence_judge": CoherenceJudgeTask,
        "instruction_follow": InstructionFollowTask,
    }
    if unit.tasks_to_run is None:
        task_names = list(all_task_classes.keys())
    else:
        task_names = [t for t in unit.tasks_to_run if t in all_task_classes]

    results: List[dict] = []
    for name in task_names:
        task = all_task_classes[name](unit.client)
        try:
            probing_results = task.run(
                corrupted_text=unit.corrupted,
                text_entry=unit.text_entry,
                granularity=unit.granularity,
                alpha=unit.alpha,
                seed=unit.seed_idx,
            )
            for r in probing_results:
                r.parsed["_inference_rep"] = unit.rep_idx
                results.append(_result_to_dict(r))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "task_name": name,
                "text_id": unit.text_entry.id,
                "granularity": unit.granularity,
                "alpha": unit.alpha,
                "seed": unit.seed_idx,
                "inference_rep": unit.rep_idx,
                "model": unit.client.model_str,
                "raw_response": "",
                "parsed": {},
                "error": str(exc),
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": 0.0,
            })
    return results


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _build_clients(
    model_name: str,
    api_key: Optional[str],
    dry_run: bool,
    inference_reps: int,
) -> List[LLMClient]:
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
# Core sweep (concurrent)
# ---------------------------------------------------------------------------

def run_experiment_concurrent(
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
    concurrency: int = 20,
) -> Dict[str, list]:
    """
    Run the full experiment sweep with concurrent LLM calls.

    Parameters
    ----------
    concurrency : int
        Maximum number of simultaneous LLM API calls.
        Recommended: 10–30 for OpenAI/Anthropic, 5–10 for Together AI.
    """
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[str, list] = defaultdict(list)

    total_per_model = (
        len(granularities) * len(alpha_levels) * len(texts) * seeds * inference_reps
    )
    total_conditions = len(models) * total_per_model

    print(
        f"\n{'='*60}\n"
        f"Concurrent experiment sweep\n"
        f"  Models          : {models}\n"
        f"  Granularities   : {granularities}\n"
        f"  Alpha levels    : {len(alpha_levels)} ({alpha_levels[0]}–{alpha_levels[-1]})\n"
        f"  Texts           : {len(texts)}\n"
        f"  Corruption seeds: {seeds}\n"
        f"  Inference reps  : {inference_reps}\n"
        f"  Concurrency     : {concurrency}\n"
        f"  Dry run         : {dry_run}\n"
        f"  Total conditions: {total_conditions}\n"
        f"{'='*60}\n"
    )

    for model_name in models:
        print(f"\n[Model: {model_name}]")

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

        # Build all work units upfront
        work_units: List[_WorkUnit] = []
        for granularity in granularities:
            corruptor = get_corruptor(granularity)
            for alpha in alpha_levels:
                for text_entry in texts:
                    for seed_idx in range(seeds):
                        corrupted = corruptor.corrupt(
                            text_entry.text, alpha=alpha, seed=seed_idx
                        )
                        for rep_idx, client in enumerate(clients):
                            work_units.append(_WorkUnit(
                                granularity=granularity,
                                alpha=alpha,
                                text_entry=text_entry,
                                seed_idx=seed_idx,
                                rep_idx=rep_idx,
                                corrupted=corrupted,
                                client=client,
                                tasks_to_run=tasks_to_run,
                            ))

        total_units = len(work_units)
        completed_count = 0
        write_lock = threading.Lock()

        print(f"  Dispatching {total_units} work units with concurrency={concurrency}...")
        t0 = time.perf_counter()

        with open(output_path, "w", encoding="utf-8") as out_fh:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_unit = {
                    executor.submit(_run_work_unit, unit): unit
                    for unit in work_units
                }

                for future in as_completed(future_to_unit):
                    result_dicts = future.result()
                    with write_lock:
                        completed_count += 1
                        for d in result_dicts:
                            out_fh.write(json.dumps(d) + "\n")
                            all_results[model_name].append(d)
                        # Progress display
                        elapsed = time.perf_counter() - t0
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        eta = (total_units - completed_count) / rate if rate > 0 else 0
                        pct = 100 * completed_count / total_units
                        bar_len = 30
                        filled = int(bar_len * completed_count / total_units)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(
                            f"\r  [{bar}] {completed_count}/{total_units} "
                            f"({pct:.1f}%)  {rate:.1f}/s  ETA {eta:.0f}s",
                            end="", flush=True
                        )

        elapsed_total = time.perf_counter() - t0
        print(f"\n  Results written to: {output_path}  ({elapsed_total:.1f}s)")

    return dict(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM comprehension phase transition experiment (concurrent)."
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help='Model strings, e.g. "openai:gpt-4o" "anthropic:claude-3-5-sonnet-20241022"'
    )
    parser.add_argument(
        "--granularities", nargs="+", default=GRANULARITIES,
        choices=["char", "word", "sentence"],
    )
    parser.add_argument(
        "--alpha-steps", type=int, default=21,
        help="Number of alpha levels (evenly spaced 0→1)"
    )
    parser.add_argument(
        "--dense-alpha", action="store_true",
        help="Also include the dense alpha grid around the critical region"
    )
    parser.add_argument(
        "--seeds", type=int, default=CORRUPTION_SEEDS,
    )
    parser.add_argument(
        "--inference-reps", type=int, default=INFERENCE_REPS,
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max simultaneous LLM API calls (default: 20)"
    )
    parser.add_argument(
        "--max-texts", type=int, default=None,
    )
    parser.add_argument(
        "--corpus-json", type=str, default=None,
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["qa", "summarization", "coherence_judge", "instruction_follow"],
        default=None,
    )
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR,
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np  # type: ignore[import]
    alpha_levels = [round(v, 2) for v in np.linspace(0.0, 1.0, args.alpha_steps).tolist()]
    if args.dense_alpha:
        alpha_levels = sorted(set(alpha_levels + ALPHA_DENSE_REGION))

    loader = CorpusLoader(
        json_path=args.corpus_json,
        max_texts=args.max_texts,
    )
    texts = loader.load()
    if not texts:
        print("ERROR: No texts loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(texts)} texts from corpus.")

    t0 = time.perf_counter()
    run_experiment_concurrent(
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
        concurrency=args.concurrency,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nExperiment complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()

# Made with Bob