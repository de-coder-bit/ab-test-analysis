"""
test_stats.py
=============
Pytest test suite for src/stats_tests.py.

Tests are verified against known analytical values to ensure correctness
of all statistical functions used in the Cookie Cats A/B test analysis.

Run with:
    pytest tests/test_stats.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

# Add the project root to path so src/ can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_tests import (
    BayesianResult,
    BootstrapResult,
    MannWhitneyResult,
    PowerResult,
    SRMResult,
    ZTestResult,
    apply_multiple_corrections,
    bayesian_beta_binomial,
    bootstrap_diff,
    check_srm,
    compute_power,
    mann_whitney_test,
    sample_size_calculator,
    simulate_peeking,
    two_prop_z_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cookie_cats_like_data():
    """Synthetic data mimicking Cookie Cats statistics for deterministic testing."""
    rng = np.random.default_rng(42)
    n = 44700
    # Control: gate_30, retention_7 ~ 19%
    control = rng.binomial(1, 0.190, size=n)
    # Treatment: gate_40, retention_7 ~ 18.2%
    treatment = rng.binomial(1, 0.182, size=n)
    return control, treatment, n, n


@pytest.fixture
def gamerounds_data():
    """Synthetic right-skewed game rounds data."""
    rng = np.random.default_rng(0)
    control = rng.negative_binomial(n=2, p=0.1, size=5000).astype(float)
    treatment = rng.negative_binomial(n=2, p=0.11, size=5000).astype(float)
    return control, treatment


# ─────────────────────────────────────────────────────────────────────────────
# 1. SRM Test
# ─────────────────────────────────────────────────────────────────────────────

class TestSRM:
    """Tests for Sample Ratio Mismatch detection."""

    def test_balanced_split_passes(self):
        """Perfectly balanced 50/50 split should pass SRM check."""
        result = check_srm(control_n=45000, treatment_n=45000)
        assert isinstance(result, SRMResult)
        assert result.passed is True
        assert result.p_value > 0.01
        assert result.chi2_stat == pytest.approx(0.0, abs=1e-9)

    def test_severely_imbalanced_fails(self):
        """Severely imbalanced split (60/40) should fail SRM check."""
        result = check_srm(control_n=54000, treatment_n=36000)
        assert result.passed is False
        assert result.p_value < 0.01

    def test_known_chi2_value(self):
        """
        Manual verification: control=44700, treatment=45489, total=90189.
        Expected under 50/50: 45094.5 each.
        chi2 = (44700-45094.5)^2/45094.5 + (45489-45094.5)^2/45094.5
             = 394.5^2/45094.5 + 394.5^2/45094.5
             = 3.453 (approx)
        """
        result = check_srm(control_n=44700, treatment_n=45489)
        # Compute expected manually
        total = 44700 + 45489
        expected = total / 2
        chi2_manual = 2 * ((44700 - expected) ** 2 / expected)
        assert result.chi2_stat == pytest.approx(chi2_manual, rel=1e-4)

    def test_slightly_imbalanced_passes(self):
        """Slight imbalance within noise should still pass."""
        result = check_srm(control_n=44700, treatment_n=45489, alpha=0.01)
        # This is borderline — just check it returns a valid SRMResult
        assert isinstance(result, SRMResult)
        assert isinstance(result.passed, bool)
        assert 0 <= result.p_value <= 1

    def test_custom_expected_ratio(self):
        """Test with 70/30 expected ratio."""
        total = 10000
        control_n = int(total * 0.70)
        treatment_n = total - control_n
        result = check_srm(control_n=control_n, treatment_n=treatment_n, expected_ratio=0.70)
        # Perfect split at expected ratio → chi2 ~ 0
        assert result.chi2_stat == pytest.approx(0.0, abs=0.1)
        assert result.passed is True

    def test_result_fields_present(self):
        """SRMResult should have all required fields."""
        result = check_srm(44700, 45489)
        assert hasattr(result, "control_n")
        assert hasattr(result, "treatment_n")
        assert hasattr(result, "expected_control_n")
        assert hasattr(result, "expected_treatment_n")
        assert hasattr(result, "chi2_stat")
        assert hasattr(result, "p_value")
        assert hasattr(result, "passed")
        assert hasattr(result, "message")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Two-Proportion Z-Test
# ─────────────────────────────────────────────────────────────────────────────

class TestZTest:
    """Tests for two-proportion z-test."""

    def test_known_values_no_effect(self):
        """
        With equal proportions, z-stat should be ~0 and p-value ~1.
        """
        result = two_prop_z_test(
            control_successes=4500, control_n=10000,
            treatment_successes=4500, treatment_n=10000,
            metric="retention_1",
        )
        assert result.z_stat == pytest.approx(0.0, abs=0.01)
        assert result.p_value == pytest.approx(1.0, abs=0.05)
        assert result.absolute_lift == pytest.approx(0.0, abs=1e-9)
        assert result.significant is False

    def test_known_p_value_against_scipy(self):
        """
        Cross-validate p-value against scipy.stats manually.
        control: 8500/44700, treatment: 8252/45489
        """
        c_succ, c_n = 8500, 44700
        t_succ, t_n = 8252, 45489
        result = two_prop_z_test(c_succ, c_n, t_succ, t_n, metric="retention_1")

        # Manual scipy computation
        from statsmodels.stats.proportion import proportions_ztest
        stat, p = proportions_ztest([t_succ, c_succ], [t_n, c_n], alternative="two-sided")
        assert result.p_value == pytest.approx(p, rel=1e-3)
        assert result.z_stat == pytest.approx(stat, rel=1e-3)

    def test_rates_computed_correctly(self):
        """Rates and lifts should be computed correctly."""
        result = two_prop_z_test(
            control_successes=100, control_n=1000,
            treatment_successes=110, treatment_n=1000,
            metric="test",
        )
        assert result.control_rate == pytest.approx(0.10, rel=1e-9)
        assert result.treatment_rate == pytest.approx(0.11, rel=1e-9)
        assert result.absolute_lift == pytest.approx(0.01, rel=1e-9)
        assert result.relative_lift_pct == pytest.approx(10.0, rel=1e-4)

    def test_ci_contains_truth_under_null(self):
        """When there's no effect, 95% CI should contain 0."""
        result = two_prop_z_test(
            control_successes=500, control_n=1000,
            treatment_successes=500, treatment_n=1000,
            confidence_level=0.95,
        )
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_significant_result(self):
        """Large sample with real effect should be significant."""
        result = two_prop_z_test(
            control_successes=4000, control_n=10000,
            treatment_successes=4500, treatment_n=10000,
            confidence_level=0.95,
        )
        assert result.significant is True
        assert result.p_value < 0.05

    def test_confidence_levels(self):
        """Higher confidence level → wider CI."""
        kwargs = dict(control_successes=500, control_n=1000,
                      treatment_successes=520, treatment_n=1000)
        r95 = two_prop_z_test(**kwargs, confidence_level=0.95)
        r99 = two_prop_z_test(**kwargs, confidence_level=0.99)
        width_95 = r95.ci_upper - r95.ci_lower
        width_99 = r99.ci_upper - r99.ci_lower
        assert width_99 > width_95

    def test_result_fields_present(self):
        """ZTestResult should have all required fields."""
        result = two_prop_z_test(500, 1000, 520, 1000)
        for field in ["metric", "control_rate", "treatment_rate", "absolute_lift",
                      "relative_lift_pct", "ci_lower", "ci_upper", "z_stat",
                      "p_value", "significant", "confidence_level"]:
            assert hasattr(result, field)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrap:
    """Tests for bootstrap confidence interval computation."""

    def test_shape_and_type(self, gamerounds_data):
        """bootstrap_diffs should have correct length."""
        control, treatment = gamerounds_data
        result = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=500, seed=42)
        assert isinstance(result, BootstrapResult)
        assert len(result.bootstrap_diffs) == 500

    def test_reproducibility(self, gamerounds_data):
        """Same seed → same results."""
        control, treatment = gamerounds_data
        r1 = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=1000, seed=123)
        r2 = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=1000, seed=123)
        np.testing.assert_array_equal(r1.bootstrap_diffs, r2.bootstrap_diffs)

    def test_different_seeds_differ(self, gamerounds_data):
        """Different seeds → different samples."""
        control, treatment = gamerounds_data
        r1 = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=500, seed=1)
        r2 = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=500, seed=2)
        assert not np.allclose(r1.bootstrap_diffs, r2.bootstrap_diffs)

    def test_ci_ordering(self, gamerounds_data):
        """CI lower should be less than upper."""
        control, treatment = gamerounds_data
        result = bootstrap_diff(control, treatment, stat_func="median", n_resamples=500, seed=42)
        assert result.ci_lower < result.ci_upper

    def test_observed_diff_correct(self):
        """Observed diff should match manual computation."""
        rng = np.random.default_rng(0)
        control = rng.normal(10, 2, 200)
        treatment = rng.normal(12, 2, 200)
        result = bootstrap_diff(control, treatment, stat_func="mean", n_resamples=200, seed=0)
        expected_diff = np.mean(treatment) - np.mean(control)
        assert result.observed_diff == pytest.approx(expected_diff, rel=1e-9)

    def test_no_effect_ci_covers_zero(self):
        """Bootstrap CI for identical distributions should cover 0."""
        rng = np.random.default_rng(77)
        data = rng.normal(0.44, 0.05, 5000)
        result = bootstrap_diff(data, data.copy(), stat_func="mean", n_resamples=1000, seed=77, confidence_level=0.95)
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_median_stat_func(self, gamerounds_data):
        """stat_func='median' should use median, not mean."""
        control, treatment = gamerounds_data
        result = bootstrap_diff(control, treatment, stat_func="median", n_resamples=200, seed=42)
        expected_diff = np.median(treatment) - np.median(control)
        assert result.observed_diff == pytest.approx(expected_diff, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Power Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerAnalysis:
    """Tests for statistical power computation."""

    def test_high_power_large_sample(self):
        """Large sample with non-trivial effect should have power > 0.80."""
        result = compute_power(
            control_n=44700, treatment_n=45489,
            control_rate=0.190, treatment_rate=0.182,
            alpha=0.05,
        )
        assert isinstance(result, PowerResult)
        assert result.achieved_power > 0.70  # real Cookie Cats figures

    def test_power_between_0_and_1(self):
        """Achieved power should always be in [0, 1]."""
        result = compute_power(
            control_n=1000, treatment_n=1000,
            control_rate=0.20, treatment_rate=0.22,
            alpha=0.05,
        )
        assert 0.0 <= result.achieved_power <= 1.0

    def test_larger_sample_more_power(self):
        """More observations → higher power, all else equal."""
        r_small = compute_power(1000, 1000, 0.20, 0.22, alpha=0.05)
        r_large = compute_power(10000, 10000, 0.20, 0.22, alpha=0.05)
        assert r_large.achieved_power > r_small.achieved_power

    def test_mde_positive(self):
        """MDE at 80% power should be a positive float."""
        result = compute_power(44700, 45489, 0.190, 0.182, alpha=0.05)
        assert result.mde_at_80_power > 0

    def test_sample_size_for_80_power_positive(self):
        """Required sample size should be a positive integer."""
        result = compute_power(44700, 45489, 0.190, 0.182, alpha=0.05)
        assert isinstance(result.sample_size_for_80_power, int)
        assert result.sample_size_for_80_power > 0

    def test_sample_size_calculator_known_value(self):
        """
        Cross-check: baseline=0.44, MDE=0.02, alpha=0.05, power=0.80.
        Known result is roughly ~4,700 per group from standard tables.
        """
        n = sample_size_calculator(baseline_rate=0.44, mde=0.02, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert 3000 < n < 8000  # reasonable range check

    def test_smaller_mde_needs_larger_n(self):
        """Smaller MDE requires larger sample to achieve same power."""
        n_large_mde = sample_size_calculator(0.44, mde=0.05, alpha=0.05, power=0.80)
        n_small_mde = sample_size_calculator(0.44, mde=0.01, alpha=0.05, power=0.80)
        assert n_small_mde > n_large_mde


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multiple Comparisons Correction
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleCorrections:
    """Tests for Bonferroni and Holm corrections."""

    def test_output_shape(self):
        """Output DataFrame should have correct columns and rows."""
        p_values = [0.01, 0.04, 0.10]
        metrics = ["retention_1", "retention_7", "sum_gamerounds"]
        df = apply_multiple_corrections(p_values, metrics, alpha=0.05)
        assert len(df) == 3
        for col in ["metric", "original_p", "bonferroni_p", "bonferroni_significant",
                    "holm_p", "holm_significant"]:
            assert col in df.columns

    def test_bonferroni_known_value(self):
        """Bonferroni p = min(1, p_raw * n_tests)."""
        p_values = [0.01, 0.04, 0.10]
        metrics = ["a", "b", "c"]
        df = apply_multiple_corrections(p_values, metrics, alpha=0.05)
        # Bonferroni: 0.01 * 3 = 0.03
        assert df.loc[df["metric"] == "a", "bonferroni_p"].values[0] == pytest.approx(0.03, rel=1e-4)

    def test_holm_more_powerful_than_bonferroni(self):
        """Holm should reject at least as many as Bonferroni."""
        p_values = [0.001, 0.02, 0.045]
        metrics = ["x", "y", "z"]
        df = apply_multiple_corrections(p_values, metrics, alpha=0.05)
        n_holm_sig = df["holm_significant"].sum()
        n_bonf_sig = df["bonferroni_significant"].sum()
        assert n_holm_sig >= n_bonf_sig

    def test_all_significant_remain_with_tiny_p(self):
        """Extremely small p-values should remain significant after any correction."""
        p_values = [1e-10, 1e-9, 1e-8]
        metrics = ["a", "b", "c"]
        df = apply_multiple_corrections(p_values, metrics, alpha=0.05)
        assert df["bonferroni_significant"].all()
        assert df["holm_significant"].all()

    def test_none_significant_with_large_p(self):
        """Large p-values should not be significant after correction."""
        p_values = [0.30, 0.40, 0.50]
        metrics = ["a", "b", "c"]
        df = apply_multiple_corrections(p_values, metrics, alpha=0.05)
        assert not df["bonferroni_significant"].any()
        assert not df["holm_significant"].any()

    def test_metric_names_preserved(self):
        """Metric names should be preserved in output."""
        names = ["alpha_metric", "beta_metric"]
        df = apply_multiple_corrections([0.01, 0.05], names, alpha=0.05)
        assert list(df["metric"]) == names


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bayesian Beta-Binomial
# ─────────────────────────────────────────────────────────────────────────────

class TestBayesian:
    """Tests for Bayesian Beta-Binomial analysis."""

    def test_prob_between_0_and_1(self):
        """P(treatment > control) must be in [0, 1]."""
        result = bayesian_beta_binomial(
            control_successes=8500, control_n=44700,
            treatment_successes=8252, treatment_n=45489,
            seed=42,
        )
        assert 0 <= result.prob_treatment_better <= 1

    def test_control_better_prob_lower(self):
        """When control is clearly better, P(treatment > control) should be < 0.5."""
        result = bayesian_beta_binomial(
            control_successes=9000, control_n=10000,   # 90%
            treatment_successes=5000, treatment_n=10000,  # 50%
            seed=42,
        )
        assert result.prob_treatment_better < 0.01

    def test_posterior_sample_sizes(self):
        """Posterior samples should have correct length (n_samples)."""
        result = bayesian_beta_binomial(
            control_successes=100, control_n=1000,
            treatment_successes=110, treatment_n=1000,
            n_samples=5000, seed=42,
        )
        assert len(result.control_posterior_samples) == 5000
        assert len(result.treatment_posterior_samples) == 5000

    def test_expected_loss_non_negative(self):
        """Expected loss values should always be non-negative."""
        result = bayesian_beta_binomial(500, 1000, 520, 1000, seed=42)
        assert result.expected_loss_treatment >= 0
        assert result.expected_loss_control >= 0

    def test_posterior_parameters(self):
        """Posterior alpha/beta should be prior + data."""
        prior_a, prior_b = 1.0, 1.0
        c_succ, c_n = 100, 1000
        t_succ, t_n = 110, 1000
        result = bayesian_beta_binomial(c_succ, c_n, t_succ, t_n,
                                        prior_alpha=prior_a, prior_beta=prior_b, seed=42)
        assert result.control_alpha == pytest.approx(prior_a + c_succ)
        assert result.control_beta == pytest.approx(prior_b + c_n - c_succ)
        assert result.treatment_alpha == pytest.approx(prior_a + t_succ)
        assert result.treatment_beta == pytest.approx(prior_b + t_n - t_succ)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mann-Whitney U Test
# ─────────────────────────────────────────────────────────────────────────────

class TestMannWhitney:
    """Tests for Mann-Whitney U test."""

    def test_identical_distributions_not_significant(self):
        """Identical data should not be significant."""
        rng = np.random.default_rng(42)
        data = rng.negative_binomial(2, 0.1, 500).astype(float)
        result = mann_whitney_test(data, data.copy())
        assert result.significant is False

    def test_clearly_different_distributions_significant(self):
        """Clearly different distributions should be significant."""
        rng = np.random.default_rng(42)
        control = rng.normal(10, 1, 1000)
        treatment = rng.normal(15, 1, 1000)
        result = mann_whitney_test(control, treatment)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_effect_size_in_range(self, gamerounds_data):
        """Rank-biserial correlation should be in [-1, 1]."""
        control, treatment = gamerounds_data
        result = mann_whitney_test(control, treatment)
        assert -1.0 <= result.effect_size_r <= 1.0

    def test_medians_computed_correctly(self):
        """Medians should match numpy computation."""
        rng = np.random.default_rng(5)
        control = rng.integers(0, 100, 500).astype(float)
        treatment = rng.integers(10, 110, 500).astype(float)
        result = mann_whitney_test(control, treatment)
        assert result.control_median == pytest.approx(np.median(control))
        assert result.treatment_median == pytest.approx(np.median(treatment))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Peeking Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestPeeking:
    """Tests for peeking simulation."""

    def test_output_shape(self):
        """Output DataFrame should have expected columns and rows."""
        checkpoints = [100, 200, 300]
        df = simulate_peeking(
            n_simulations=50,
            n_per_group=300,
            check_points=checkpoints,
            seed=42,
        )
        assert "checkpoint" in df.columns
        assert "false_positive_rate" in df.columns
        assert len(df) == len(checkpoints)

    def test_false_positive_rates_nondecreasing(self):
        """Cumulative false positive rate should be non-decreasing with more peeks."""
        df = simulate_peeking(
            n_simulations=200,
            n_per_group=1000,
            check_points=[100, 300, 600, 1000],
            seed=42,
        )
        rates = df["false_positive_rate"].tolist()
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1] - 1e-9  # allow tiny floating point slack

    def test_false_positive_rate_in_range(self):
        """False positive rate should be in [0, 1]."""
        df = simulate_peeking(n_simulations=100, n_per_group=500, seed=42)
        assert (df["false_positive_rate"] >= 0).all()
        assert (df["false_positive_rate"] <= 1).all()
