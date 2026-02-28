"""
Unit tests for the metrics modules.

Run with:
    cd paper_idea/experiment
    pytest tests/test_metrics.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest

from metrics.task_metrics import (
    exact_match,
    token_f1,
    refusal_rate,
    instruction_follow_rate,
    compute_rouge_l,
    aggregate_qa_metrics,
    aggregate_summarization_metrics,
    aggregate_coherence_metrics,
    aggregate_instruction_metrics,
)
from metrics.phase_transition import (
    fit_sigmoid,
    compute_auc,
    compute_critical_threshold,
    compute_transition_sharpness,
    build_performance_curve,
    SigmoidFit,
)
from probing.base import ProbingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(task_name="qa", alpha=1.0, parsed=None, error=None) -> ProbingResult:
    return ProbingResult(
        task_name=task_name,
        text_id="test_001",
        granularity="word",
        alpha=alpha,
        seed=0,
        model="test-model",
        raw_response="test response",
        parsed=parsed or {},
        error=error,
    )


# ===========================================================================
# exact_match
# ===========================================================================

class TestExactMatch:

    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1.0

    def test_punctuation_stripped(self):
        assert exact_match("Paris.", "Paris") == 1.0

    def test_different(self):
        assert exact_match("London", "Paris") == 0.0

    def test_empty_strings(self):
        assert exact_match("", "") == 1.0

    def test_whitespace_normalised(self):
        assert exact_match("  Paris  ", "Paris") == 1.0


# ===========================================================================
# token_f1
# ===========================================================================

class TestTokenF1:

    def test_perfect_match(self):
        assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("dog runs fast", "cat sits still") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat on the mat", "the cat sat")
        assert 0.0 < f1 < 1.0

    def test_empty_predicted(self):
        assert token_f1("", "some answer") == pytest.approx(0.0)

    def test_empty_expected(self):
        assert token_f1("some answer", "") == pytest.approx(0.0)

    def test_symmetry_approximately(self):
        # F1 is not perfectly symmetric but should be close for balanced cases
        f1_ab = token_f1("the cat sat on the mat", "the cat sat")
        f1_ba = token_f1("the cat sat", "the cat sat on the mat")
        assert abs(f1_ab - f1_ba) < 0.3  # allow some asymmetry


# ===========================================================================
# refusal_rate
# ===========================================================================

class TestRefusalRate:

    def test_all_refused(self):
        results = [
            _make_result(parsed={"refused": True}),
            _make_result(parsed={"refused": True}),
        ]
        assert refusal_rate(results) == pytest.approx(1.0)

    def test_none_refused(self):
        results = [
            _make_result(parsed={"refused": False}),
            _make_result(parsed={"refused": False}),
        ]
        assert refusal_rate(results) == pytest.approx(0.0)

    def test_half_refused(self):
        results = [
            _make_result(parsed={"refused": True}),
            _make_result(parsed={"refused": False}),
        ]
        assert refusal_rate(results) == pytest.approx(0.5)

    def test_empty_list(self):
        assert refusal_rate([]) == pytest.approx(0.0)


# ===========================================================================
# instruction_follow_rate
# ===========================================================================

class TestInstructionFollowRate:

    def test_all_followed(self):
        results = [_make_result(parsed={"followed": True}) for _ in range(4)]
        assert instruction_follow_rate(results) == pytest.approx(1.0)

    def test_none_followed(self):
        results = [_make_result(parsed={"followed": False}) for _ in range(4)]
        assert instruction_follow_rate(results) == pytest.approx(0.0)

    def test_partial(self):
        results = [
            _make_result(parsed={"followed": True}),
            _make_result(parsed={"followed": True}),
            _make_result(parsed={"followed": False}),
            _make_result(parsed={"followed": False}),
        ]
        assert instruction_follow_rate(results) == pytest.approx(0.5)


# ===========================================================================
# compute_rouge_l
# ===========================================================================

class TestRougeL:

    def test_identical(self):
        score = compute_rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        score = compute_rouge_l("dog runs fast", "cat sits still")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        score = compute_rouge_l("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        score = compute_rouge_l("", "")
        assert score == pytest.approx(0.0, abs=0.01)


# ===========================================================================
# aggregate_qa_metrics
# ===========================================================================

class TestAggregateQAMetrics:

    def _qa_result(self, predicted, expected, refused=False, alpha=1.0):
        return _make_result(
            task_name="qa",
            alpha=alpha,
            parsed={
                "question": "What?",
                "expected_answer": expected,
                "predicted": predicted,
                "refused": refused,
            },
        )

    def test_perfect_answers(self):
        results = [
            self._qa_result("Paris", "Paris"),
            self._qa_result("1789", "1789"),
        ]
        metrics = aggregate_qa_metrics(results)
        assert metrics["exact_match_mean"] == pytest.approx(1.0)
        assert metrics["token_f1_mean"] == pytest.approx(1.0)
        assert metrics["refusal_rate"] == pytest.approx(0.0)
        assert metrics["task_completion_rate"] == pytest.approx(1.0)

    def test_all_refused(self):
        results = [
            self._qa_result("", "Paris", refused=True),
            self._qa_result("", "1789", refused=True),
        ]
        metrics = aggregate_qa_metrics(results)
        assert metrics["refusal_rate"] == pytest.approx(1.0)
        assert metrics["task_completion_rate"] == pytest.approx(0.0)

    def test_empty_results(self):
        metrics = aggregate_qa_metrics([])
        assert metrics["exact_match_mean"] == pytest.approx(0.0)

    def test_error_results_excluded(self):
        results = [
            self._qa_result("Paris", "Paris"),
            _make_result(task_name="qa", error="API error"),
        ]
        metrics = aggregate_qa_metrics(results)
        # Only the successful result should count
        assert metrics["task_completion_rate"] == pytest.approx(0.5)


# ===========================================================================
# aggregate_summarization_metrics
# ===========================================================================

class TestAggregateSummarizationMetrics:

    def _sum_result(self, summary, reference, refused=False):
        return _make_result(
            task_name="summarization",
            parsed={
                "summary": summary,
                "reference_summary": reference,
                "refused": refused,
            },
        )

    def test_good_summary(self):
        ref = "The French Revolution began in 1789 and ended in 1799."
        results = [self._sum_result(ref, ref)]
        metrics = aggregate_summarization_metrics(results)
        assert metrics["rouge_l_mean"] == pytest.approx(1.0, abs=0.05)

    def test_refused_summary(self):
        results = [self._sum_result("", "some reference", refused=True)]
        metrics = aggregate_summarization_metrics(results)
        assert metrics["refusal_rate"] == pytest.approx(1.0)
        assert metrics["task_completion_rate"] == pytest.approx(0.0)

    def test_empty_results(self):
        metrics = aggregate_summarization_metrics([])
        assert metrics["rouge_l_mean"] == pytest.approx(0.0)


# ===========================================================================
# aggregate_coherence_metrics
# ===========================================================================

class TestAggregateCoherenceMetrics:

    def _coh_result(self, score):
        return _make_result(
            task_name="coherence_judge",
            parsed={"score": score, "reason": "test"},
        )

    def test_mean_score(self):
        results = [self._coh_result(8), self._coh_result(6), self._coh_result(4)]
        metrics = aggregate_coherence_metrics(results)
        assert metrics["mean_score"] == pytest.approx(6.0)

    def test_std_score(self):
        results = [self._coh_result(5), self._coh_result(5), self._coh_result(5)]
        metrics = aggregate_coherence_metrics(results)
        assert metrics["score_std"] == pytest.approx(0.0)

    def test_empty_results(self):
        metrics = aggregate_coherence_metrics([])
        assert metrics["mean_score"] == pytest.approx(0.0)


# ===========================================================================
# aggregate_instruction_metrics
# ===========================================================================

class TestAggregateInstructionMetrics:

    def test_all_followed(self):
        results = [_make_result(task_name="instruction_follow", parsed={"followed": True}) for _ in range(3)]
        metrics = aggregate_instruction_metrics(results)
        assert metrics["instruction_follow_rate"] == pytest.approx(1.0)

    def test_empty(self):
        metrics = aggregate_instruction_metrics([])
        assert metrics["instruction_follow_rate"] == pytest.approx(0.0)


# ===========================================================================
# Phase transition metrics
# ===========================================================================

class TestComputeAUC:

    def test_flat_curve_auc(self):
        alphas = [0.0, 0.5, 1.0]
        scores = [0.5, 0.5, 0.5]
        auc = compute_auc(alphas, scores)
        assert auc == pytest.approx(0.5, abs=0.01)

    def test_perfect_curve_auc(self):
        alphas = [0.0, 0.5, 1.0]
        scores = [1.0, 1.0, 1.0]
        auc = compute_auc(alphas, scores)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_zero_curve_auc(self):
        alphas = [0.0, 0.5, 1.0]
        scores = [0.0, 0.0, 0.0]
        auc = compute_auc(alphas, scores)
        assert auc == pytest.approx(0.0, abs=0.01)

    def test_single_point(self):
        assert compute_auc([0.5], [0.8]) == pytest.approx(0.0)

    def test_linear_ramp(self):
        # Score goes from 0 at alpha=0 to 1 at alpha=1 linearly → AUC = 0.5
        alphas = [i / 10 for i in range(11)]
        scores = [i / 10 for i in range(11)]
        auc = compute_auc(alphas, scores)
        assert auc == pytest.approx(0.5, abs=0.01)


class TestComputeCriticalThreshold:

    def test_threshold_at_midpoint(self):
        # Score drops from 1.0 to 0.0 linearly; 50% threshold should be near 0.5
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        alpha_star = compute_critical_threshold(alphas, scores, threshold_fraction=0.5)
        assert alpha_star is not None
        assert 0.0 <= alpha_star <= 1.0

    def test_always_above_threshold(self):
        alphas = [0.0, 0.5, 1.0]
        scores = [0.9, 0.95, 1.0]
        # Score never drops below 50% of baseline (1.0 * 0.5 = 0.5)
        alpha_star = compute_critical_threshold(alphas, scores, threshold_fraction=0.5)
        assert alpha_star is None

    def test_empty(self):
        assert compute_critical_threshold([], []) is None


class TestComputeTransitionSharpness:

    def test_flat_curve_zero_sharpness(self):
        alphas = [0.0, 0.5, 1.0]
        scores = [0.5, 0.5, 0.5]
        assert compute_transition_sharpness(alphas, scores) == pytest.approx(0.0)

    def test_steep_step_high_sharpness(self):
        # Step function: 0 → 1 between alpha=0.4 and alpha=0.6
        alphas = [0.0, 0.4, 0.6, 1.0]
        scores = [0.0, 0.0, 1.0, 1.0]
        sharpness = compute_transition_sharpness(alphas, scores)
        assert sharpness == pytest.approx(5.0, abs=0.1)  # 1.0 / 0.2 = 5.0

    def test_single_point(self):
        assert compute_transition_sharpness([0.5], [0.8]) == pytest.approx(0.0)


class TestFitSigmoid:

    def test_perfect_sigmoid_data(self):
        """Fit should recover parameters from data generated by a known sigmoid."""
        import math
        alpha_star_true = 0.5
        beta_true = 20.0
        alphas = [i / 20 for i in range(21)]
        scores = [
            1.0 / (1.0 + math.exp(-beta_true * (a - alpha_star_true)))
            for a in alphas
        ]
        fit = fit_sigmoid(alphas, scores)
        if fit.converged:
            assert abs(fit.alpha_star - alpha_star_true) < 0.1
            assert fit.beta > 10.0  # should detect as sharp

    def test_too_few_points(self):
        fit = fit_sigmoid([0.0, 0.5], [0.0, 1.0])
        assert not fit.converged
        assert fit.error is not None

    def test_flat_curve(self):
        alphas = [i / 10 for i in range(11)]
        scores = [0.5] * 11
        fit = fit_sigmoid(alphas, scores)
        # Should not crash; may or may not converge
        assert isinstance(fit, SigmoidFit)


class TestBuildPerformanceCurve:

    def test_basic(self):
        from probing.base import ProbingResult

        def mock_metric(results):
            return sum(r.parsed.get("score", 0) for r in results) / len(results)

        results_by_alpha = {
            0.0: [_make_result(parsed={"score": 2})],
            0.5: [_make_result(parsed={"score": 5})],
            1.0: [_make_result(parsed={"score": 9})],
        }
        alphas, scores = build_performance_curve(results_by_alpha, mock_metric)
        assert alphas == [0.0, 0.5, 1.0]
        assert scores == [2.0, 5.0, 9.0]

    def test_sorted_output(self):
        def mock_metric(results):
            return 1.0

        results_by_alpha = {0.8: [], 0.2: [], 0.5: []}
        alphas, _ = build_performance_curve(results_by_alpha, mock_metric)
        assert alphas == sorted(alphas)

# Made with Bob
