"""
Task-level evaluation metrics.

All functions operate on lists of ProbingResult objects and return
plain dicts / floats so they have no dependency on heavy ML libraries
at import time.  BERTScore and ROUGE are imported lazily.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Any

from probing.base import ProbingResult


# ---------------------------------------------------------------------------
# Low-level string utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 if normalized strings match, else 0.0."""
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


def token_f1(predicted: str, expected: str) -> float:
    """
    Token-level F1 between predicted and expected answer strings.
    Standard SQuAD-style metric.
    """
    pred_tokens = _normalize(predicted).split()
    exp_tokens = _normalize(expected).split()

    if not pred_tokens or not exp_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(exp_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(exp_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def refusal_rate(results: List[ProbingResult]) -> float:
    """Fraction of results where the LLM refused to answer."""
    if not results:
        return 0.0
    refused = sum(1 for r in results if r.parsed.get("refused", False))
    return refused / len(results)


def instruction_follow_rate(results: List[ProbingResult]) -> float:
    """Fraction of instruction-follow results where the keyword was present."""
    if not results:
        return 0.0
    followed = sum(1 for r in results if r.parsed.get("followed", False))
    return followed / len(results)


# ---------------------------------------------------------------------------
# ROUGE-L (lazy import)
# ---------------------------------------------------------------------------

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis and reference.
    Returns 0.0 if rouge_score is not installed.
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore[import]
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except ImportError:
        return _rouge_l_fallback(hypothesis, reference)


def _rouge_l_fallback(hypothesis: str, reference: str) -> float:
    """Pure-Python LCS-based ROUGE-L fallback (no external deps)."""
    h_tokens = _normalize(hypothesis).split()
    r_tokens = _normalize(reference).split()
    lcs_len = _lcs_length(h_tokens, r_tokens)
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / len(h_tokens) if h_tokens else 0.0
    recall = lcs_len / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of lists a and b."""
    m, n = len(a), len(b)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# BERTScore (lazy import)
# ---------------------------------------------------------------------------

def compute_bert_score(hypothesis: str, reference: str, lang: str = "en") -> float:
    """
    Compute BERTScore F1 between hypothesis and reference.
    Returns ROUGE-L as a fallback if bert_score is not installed.
    """
    try:
        from bert_score import score as bert_score_fn  # type: ignore[import]
        P, R, F1 = bert_score_fn(
            [hypothesis], [reference], lang=lang, verbose=False
        )
        return float(F1[0])
    except ImportError:
        # Graceful degradation: use ROUGE-L
        return compute_rouge_l(hypothesis, reference)


# ---------------------------------------------------------------------------
# Aggregation helpers — operate on lists of ProbingResult
# ---------------------------------------------------------------------------

def aggregate_qa_metrics(results: List[ProbingResult]) -> Dict[str, float]:
    """
    Aggregate QA metrics across a list of QA ProbingResults.

    Returns dict with keys:
      exact_match_mean, token_f1_mean, refusal_rate, task_completion_rate
    """
    if not results:
        return {"exact_match_mean": 0.0, "token_f1_mean": 0.0,
                "refusal_rate": 0.0, "task_completion_rate": 0.0}

    em_scores: List[float] = []
    f1_scores: List[float] = []
    completed = 0

    for r in results:
        if not r.ok:
            continue
        predicted = r.parsed.get("predicted", "")
        expected = r.parsed.get("expected_answer", "")
        refused = r.parsed.get("refused", False)

        if not refused and predicted:
            completed += 1
            em_scores.append(exact_match(predicted, expected))
            f1_scores.append(token_f1(predicted, expected))

    n = len(results)
    return {
        "exact_match_mean": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "token_f1_mean": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "refusal_rate": refusal_rate(results),
        "task_completion_rate": completed / n if n else 0.0,
    }


def aggregate_summarization_metrics(results: List[ProbingResult]) -> Dict[str, float]:
    """
    Aggregate summarization metrics.

    Returns dict with keys:
      rouge_l_mean, bert_score_mean, refusal_rate, task_completion_rate
    """
    if not results:
        return {"rouge_l_mean": 0.0, "bert_score_mean": 0.0,
                "refusal_rate": 0.0, "task_completion_rate": 0.0}

    rouge_scores: List[float] = []
    bert_scores: List[float] = []
    completed = 0

    for r in results:
        if not r.ok:
            continue
        summary = r.parsed.get("summary", "")
        reference = r.parsed.get("reference_summary", "")
        refused = r.parsed.get("refused", False)

        if not refused and summary and reference:
            completed += 1
            rouge_scores.append(compute_rouge_l(summary, reference))
            bert_scores.append(compute_bert_score(summary, reference))

    n = len(results)
    return {
        "rouge_l_mean": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "bert_score_mean": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
        "refusal_rate": refusal_rate(results),
        "task_completion_rate": completed / n if n else 0.0,
    }


def aggregate_coherence_metrics(results: List[ProbingResult]) -> Dict[str, float]:
    """
    Aggregate coherence judge metrics.

    Returns dict with keys:
      mean_score, score_std, task_completion_rate
    """
    if not results:
        return {"mean_score": 0.0, "score_std": 0.0, "task_completion_rate": 0.0}

    scores: List[float] = []
    for r in results:
        if r.ok and r.parsed.get("score") is not None:
            scores.append(float(r.parsed["score"]))

    n = len(results)
    if not scores:
        return {"mean_score": 0.0, "score_std": 0.0, "task_completion_rate": 0.0}

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5

    return {
        "mean_score": mean,
        "score_std": std,
        "task_completion_rate": len(scores) / n,
    }


def aggregate_instruction_metrics(results: List[ProbingResult]) -> Dict[str, float]:
    """
    Aggregate instruction-follow metrics.

    Returns dict with keys:
      instruction_follow_rate, task_completion_rate
    """
    if not results:
        return {"instruction_follow_rate": 0.0, "task_completion_rate": 0.0}

    completed = sum(1 for r in results if r.ok)
    return {
        "instruction_follow_rate": instruction_follow_rate(results),
        "task_completion_rate": completed / len(results),
    }

# Made with Bob
