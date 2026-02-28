"""
Post-hoc analysis and plotting.

Reads the JSONL result files produced by runner.py, computes per-alpha
aggregate metrics (mean ± std / 95% CI across inference repetitions),
fits sigmoid curves, and generates plots.

Usage
-----
python analysis.py --results-dir results/
python analysis.py --results-dir results/ --no-plots   # metrics only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from metrics.task_metrics import (
    aggregate_qa_metrics,
    aggregate_summarization_metrics,
    aggregate_coherence_metrics,
    aggregate_instruction_metrics,
)
from metrics.phase_transition import (
    SigmoidFit,
    fit_sigmoid,
    compute_auc,
    compute_critical_threshold,
    compute_transition_sharpness,
    build_performance_curve,
)
from probing.base import ProbingResult
from config import PLOTS_DIR


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> Dict[str, List[dict]]:
    """
    Load all JSONL result files from results_dir.
    Returns {model_name: [result_dict, ...]}
    """
    model_results: Dict[str, List[dict]] = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".jsonl"):
            continue
        model_name = fname.replace(".jsonl", "").replace("_", "/", 1)
        path = os.path.join(results_dir, fname)
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  [WARN] Skipping malformed JSON at {fname}:{lineno}: {exc}")
        if not records:
            print(f"  Skipping '{model_name}' — no records (file may still be writing)")
            continue
        model_results[model_name] = records
        print(f"  Loaded {len(records):,} records for model '{model_name}'")
    return model_results


def _dict_to_probing_result(d: dict) -> ProbingResult:
    return ProbingResult(
        task_name=d["task_name"],
        text_id=d["text_id"],
        granularity=d["granularity"],
        alpha=d["alpha"],
        seed=d["seed"],
        model=d["model"],
        raw_response=d["raw_response"],
        parsed=d.get("parsed", {}),
        error=d.get("error"),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        latency_seconds=d.get("latency_seconds", 0.0),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_TASK_AGGREGATORS = {
    "qa": aggregate_qa_metrics,
    "summarization": aggregate_summarization_metrics,
    "coherence_judge": aggregate_coherence_metrics,
    "instruction_follow": aggregate_instruction_metrics,
}

_PRIMARY_METRIC = {
    "qa": "token_f1_mean",
    "summarization": "rouge_l_mean",
    "coherence_judge": "mean_score",
    "instruction_follow": "instruction_follow_rate",
}


# ---------------------------------------------------------------------------
# Dispersion helpers
# ---------------------------------------------------------------------------

def _mean_std_ci(values: List[float]) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, ci_low, ci_high) for a list of values.
    95% CI uses the normal approximation: mean ± 1.96 * std / sqrt(n).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    margin = 1.96 * std / (n ** 0.5) if n > 1 else 0.0
    return mean, std, mean - margin, mean + margin


def compute_performance_curves(
    records: List[dict],
    granularity: Optional[str] = None,
) -> Dict[str, Tuple[List[float], List[float], List[float], List[float], List[float]]]:
    """
    Compute performance curves for each task, with dispersion across inference reps.

    Parameters
    ----------
    records     : flat list of result dicts for one model
    granularity : if given, filter to this granularity only

    Returns
    -------
    {task_name: (alphas, means, stds, ci_lows, ci_highs)}

    Dispersion is computed by:
      1. Grouping results by (alpha, inference_rep)
      2. Computing the primary metric for each rep independently
      3. Reporting mean ± std / CI across reps at each alpha
    """
    # Group by (task_name, alpha, inference_rep)
    grouped: Dict[str, Dict[float, Dict[int, List[ProbingResult]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for d in records:
        if granularity and d["granularity"] != granularity:
            continue
        task = d["task_name"]
        alpha = d["alpha"]
        rep = d.get("inference_rep", 0)
        grouped[task][alpha][rep].append(_dict_to_probing_result(d))

    curves: Dict[str, Tuple[List[float], List[float], List[float], List[float], List[float]]] = {}

    for task_name, by_alpha in grouped.items():
        agg_fn = _TASK_AGGREGATORS.get(task_name)
        primary_key = _PRIMARY_METRIC.get(task_name, "task_completion_rate")
        if agg_fn is None:
            continue

        sorted_alphas = sorted(by_alpha.keys())
        means, stds, ci_lows, ci_highs = [], [], [], []

        for alpha in sorted_alphas:
            by_rep = by_alpha[alpha]
            rep_scores = [
                agg_fn(results).get(primary_key, 0.0)
                for results in by_rep.values()
                if results
            ]
            m, s, lo, hi = _mean_std_ci(rep_scores)
            means.append(m)
            stds.append(s)
            ci_lows.append(lo)
            ci_highs.append(hi)

        curves[task_name] = (sorted_alphas, means, stds, ci_lows, ci_highs)

    return curves


# ---------------------------------------------------------------------------
# Phase transition summary
# ---------------------------------------------------------------------------

def summarise_phase_transition(
    curves: Dict[str, Tuple[List[float], List[float], List[float], List[float], List[float]]],
    model: str,
    granularity: str,
) -> List[dict]:
    """
    For each task curve, fit a sigmoid and compute transition metrics.
    Returns a list of summary dicts including dispersion statistics.
    """
    summaries = []
    for task_name, (alphas, means, stds, ci_lows, ci_highs) in curves.items():
        if len(alphas) < 4:
            continue
        fit = fit_sigmoid(alphas, means)
        auc = compute_auc(alphas, means)
        alpha_star = compute_critical_threshold(alphas, means)
        sharpness = compute_transition_sharpness(alphas, means)

        # Mean std across alpha levels (overall dispersion)
        mean_std = sum(stds) / len(stds) if stds else 0.0

        summaries.append({
            "model": model,
            "granularity": granularity,
            "task": task_name,
            "alpha_star_50pct": alpha_star,
            "sigmoid_alpha_star": fit.alpha_star,
            "sigmoid_beta": fit.beta,
            "sigmoid_r_squared": fit.r_squared,
            "sigmoid_converged": fit.converged,
            "is_sharp_transition": fit.is_sharp,
            "auc": auc,
            "max_slope": sharpness,
            "mean_std_across_reps": mean_std,
            "per_alpha_means": dict(zip(alphas, means)),
            "per_alpha_stds": dict(zip(alphas, stds)),
            "per_alpha_ci_low": dict(zip(alphas, ci_lows)),
            "per_alpha_ci_high": dict(zip(alphas, ci_highs)),
        })
    return summaries


# ---------------------------------------------------------------------------
# Plotting (matplotlib, optional)
# ---------------------------------------------------------------------------

def plot_curves(
    curves: Dict[str, Tuple[List[float], List[float], List[float], List[float], List[float]]],
    model: str,
    granularity: str,
    plots_dir: str,
    sigmoid_fits: Optional[Dict[str, SigmoidFit]] = None,
) -> None:
    """
    Plot M(alpha) curves with ±1 std shaded band and optional sigmoid fit overlay.
    Silently skips if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        print("  [WARN] matplotlib not installed; skipping plots.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    task_order = ["qa", "summarization", "coherence_judge", "instruction_follow"]
    task_labels = {
        "qa": "QA (Token F1)",
        "summarization": "Summarization (ROUGE-L)",
        "coherence_judge": "Coherence Self-Rating (mean/10)",
        "instruction_follow": "Instruction Follow Rate",
    }

    for ax, task_name in zip(axes, task_order):
        if task_name not in curves:
            ax.set_visible(False)
            continue

        alphas, means, stds, ci_lows, ci_highs = curves[task_name]
        alphas_arr = np.array(alphas)
        means_arr = np.array(means)
        stds_arr = np.array(stds)

        # Mean line
        ax.plot(alphas_arr, means_arr, "o-", color="steelblue", linewidth=2,
                markersize=5, label="Mean across reps")

        # ±1 std shaded band
        if stds_arr.max() > 0:
            ax.fill_between(
                alphas_arr,
                means_arr - stds_arr,
                means_arr + stds_arr,
                alpha=0.2, color="steelblue", label="±1 std"
            )

        # 95% CI error bars
        yerr_low = means_arr - np.array(ci_lows)
        yerr_high = np.array(ci_highs) - means_arr
        ax.errorbar(
            alphas_arr, means_arr,
            yerr=[yerr_low, yerr_high],
            fmt="none", color="steelblue", alpha=0.5, capsize=3,
        )

        # Sigmoid fit overlay
        if sigmoid_fits and task_name in sigmoid_fits:
            fit = sigmoid_fits[task_name]
            if fit.converged:
                x_fine = np.linspace(0, 1, 200)
                from metrics.phase_transition import _sigmoid_vec
                y_fine = _sigmoid_vec(
                    list(x_fine), fit.alpha_star, fit.beta, fit.L, fit.offset
                )
                ax.plot(x_fine, y_fine, "--", color="tomato", linewidth=1.5,
                        label=f"Sigmoid fit (β={fit.beta:.1f}, α*={fit.alpha_star:.2f})")
                ax.axvline(fit.alpha_star, color="tomato", linestyle=":", alpha=0.6)

        ax.set_xlabel("Coherence ratio α")
        ax.set_ylabel(task_labels.get(task_name, task_name))
        ax.set_title(task_labels.get(task_name, task_name))
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Model: {model}  |  Granularity: {granularity}", fontsize=13)
    plt.tight_layout()

    safe_model = model.replace("/", "_").replace(":", "_")
    fname = os.path.join(plots_dir, f"{safe_model}_{granularity}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse experiment results and generate plots."
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing JSONL result files from runner.py"
    )
    parser.add_argument(
        "--plots-dir", type=str, default=PLOTS_DIR,
        help="Directory to save plots (default: plots/)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--granularities", nargs="+",
        default=["char", "word", "sentence"],
        help="Granularities to analyse"
    )
    parser.add_argument(
        "--summary-out", type=str, default=None,
        help="Path to write JSON summary of phase transition metrics"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading results from: {args.results_dir}")
    model_records = load_results(args.results_dir)

    if not model_records:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    all_summaries = []

    for model, records in model_records.items():
        print(f"\n{'─'*50}")
        print(f"Model: {model}  ({len(records):,} total records)")

        for granularity in args.granularities:
            print(f"\n  Granularity: {granularity}")
            curves = compute_performance_curves(records, granularity=granularity)

            if not curves:
                print("    No data for this granularity.")
                continue

            # Fit sigmoids
            sigmoid_fits = {}
            for task_name, (alphas, means, stds, ci_lows, ci_highs) in curves.items():
                if len(alphas) >= 4:
                    sigmoid_fits[task_name] = fit_sigmoid(alphas, means)

            # Print summary table
            print(f"    {'Task':<22} {'α*':>6} {'β':>8} {'R²':>6} {'AUC':>6} {'Std':>6} {'Sharp?':>7}")
            print(f"    {'─'*22} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")
            for task_name, (alphas, means, stds, ci_lows, ci_highs) in curves.items():
                fit = sigmoid_fits.get(task_name)
                auc = compute_auc(alphas, means)
                mean_std = sum(stds) / len(stds) if stds else 0.0
                if fit:
                    print(
                        f"    {task_name:<22} "
                        f"{fit.alpha_star:>6.2f} "
                        f"{fit.beta:>8.2f} "
                        f"{fit.r_squared:>6.3f} "
                        f"{auc:>6.3f} "
                        f"{mean_std:>6.3f} "
                        f"{'YES' if fit.is_sharp else 'no':>7}"
                    )

            # Collect summaries
            summaries = summarise_phase_transition(curves, model, granularity)
            all_summaries.extend(summaries)

            # Plot
            if not args.no_plots:
                plot_curves(
                    curves=curves,
                    model=model,
                    granularity=granularity,
                    plots_dir=args.plots_dir,
                    sigmoid_fits=sigmoid_fits,
                )

    # Write summary JSON
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, indent=2)
        print(f"\nSummary written to: {args.summary_out}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()

# Made with Bob
