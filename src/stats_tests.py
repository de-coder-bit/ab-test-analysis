"""
stats_tests.py
==============
Statistical testing utilities for the Cookie Cats mobile game A/B test analysis.

Provides:
- Sample-Ratio Mismatch (SRM) detection
- Two-proportion z-test
- Mann-Whitney U test
- Multiple-testing corrections (Bonferroni / Holm)
- Bootstrap confidence-interval estimation
- Bayesian Beta-Binomial inference
- Power analysis
- Peeking simulation
- Sample-size calculator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import (
    proportion_confint,
    proportions_ztest,
    proportion_effectsize,
)
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class SRMResult:
    """Result of a Sample-Ratio Mismatch chi-square test."""

    control_n: int
    treatment_n: int
    expected_control_n: float
    expected_treatment_n: float
    chi2_stat: float
    p_value: float
    passed: bool  # True if p > alpha (no SRM detected)
    message: str


@dataclass
class ZTestResult:
    """Result of a two-proportion z-test."""

    metric: str
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift_pct: float
    ci_lower: float
    ci_upper: float
    z_stat: float
    p_value: float
    significant: bool
    confidence_level: float


@dataclass
class MannWhitneyResult:
    """Result of a Mann-Whitney U test."""

    metric: str
    control_median: float
    treatment_median: float
    absolute_diff: float
    relative_diff_pct: float
    u_stat: float
    p_value: float
    effect_size_r: float  # rank-biserial correlation
    significant: bool


@dataclass
class BootstrapResult:
    """Result of a bootstrap difference estimation."""

    metric: str
    observed_diff: float
    ci_lower: float
    ci_upper: float
    bootstrap_diffs: np.ndarray
    n_resamples: int
    seed: int


@dataclass
class BayesianResult:
    """Result of a Bayesian Beta-Binomial analysis."""

    metric: str
    control_alpha: float
    control_beta: float
    treatment_alpha: float
    treatment_beta: float
    prob_treatment_better: float
    expected_loss_treatment: float  # expected loss if you pick treatment and it's wrong
    expected_loss_control: float    # expected loss if you pick control and it's wrong
    control_posterior_samples: np.ndarray
    treatment_posterior_samples: np.ndarray


@dataclass
class PowerResult:
    """Result of a power / sample-size analysis."""

    metric: str
    n_control: int
    n_treatment: int
    observed_effect_size: float
    achieved_power: float
    alpha: float
    mde_at_80_power: float       # minimum detectable effect at 80 % power
    sample_size_for_80_power: int  # per group


# ---------------------------------------------------------------------------
# 1. Sample-Ratio Mismatch
# ---------------------------------------------------------------------------


def check_srm(
    control_n: int,
    treatment_n: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01,
) -> SRMResult:
    """Detect a Sample-Ratio Mismatch (SRM) using a chi-square test.

    Parameters
    ----------
    control_n:
        Number of users observed in the control group.
    treatment_n:
        Number of users observed in the treatment group.
    expected_ratio:
        Expected proportion of users in the control group (default 0.5 for a
        50/50 split).
    alpha:
        Significance level for the SRM test (default 0.01).

    Returns
    -------
    SRMResult
        Dataclass containing the test statistic, p-value, and a human-readable
        message.
    """
    total = control_n + treatment_n
    expected_control = total * expected_ratio
    expected_treatment = total * (1.0 - expected_ratio)

    observed = np.array([control_n, treatment_n], dtype=float)
    expected = np.array([expected_control, expected_treatment], dtype=float)

    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    passed = bool(p_value > alpha)

    if passed:
        message = (
            f"No SRM detected (χ²={chi2_stat:.4f}, p={p_value:.4f}). "
            f"Observed split ({control_n}/{treatment_n}) is consistent with "
            f"expected {expected_ratio:.0%}/{1-expected_ratio:.0%} ratio."
        )
    else:
        message = (
            f"⚠️  SRM DETECTED (χ²={chi2_stat:.4f}, p={p_value:.4f}). "
            f"Observed split ({control_n}/{treatment_n}) differs significantly "
            f"from expected {expected_ratio:.0%}/{1-expected_ratio:.0%} ratio. "
            "Investigate data pipeline before drawing conclusions."
        )

    return SRMResult(
        control_n=control_n,
        treatment_n=treatment_n,
        expected_control_n=expected_control,
        expected_treatment_n=expected_treatment,
        chi2_stat=float(chi2_stat),
        p_value=float(p_value),
        passed=passed,
        message=message,
    )


# ---------------------------------------------------------------------------
# 2. Two-proportion z-test
# ---------------------------------------------------------------------------


def two_prop_z_test(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    metric: str = "retention",
    confidence_level: float = 0.95,
) -> ZTestResult:
    """Run a two-proportion z-test and compute a CI for the difference.

    Parameters
    ----------
    control_successes:
        Number of successes (e.g. retained users) in the control group.
    control_n:
        Total users in the control group.
    treatment_successes:
        Number of successes in the treatment group.
    treatment_n:
        Total users in the treatment group.
    metric:
        Label for the metric being tested.
    confidence_level:
        Desired confidence level for the difference CI (default 0.95).

    Returns
    -------
    ZTestResult
        Dataclass with rates, lift, CI, z-statistic, p-value, and significance.
    """
    control_rate = control_successes / control_n
    treatment_rate = treatment_successes / treatment_n

    # Two-sided z-test using statsmodels
    count = np.array([treatment_successes, control_successes])
    nobs = np.array([treatment_n, control_n])
    z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")

    # CI of the difference using normal approximation
    z_critical = stats.norm.ppf((1.0 + confidence_level) / 2.0)
    diff_se = np.sqrt(
        control_rate * (1.0 - control_rate) / control_n
        + treatment_rate * (1.0 - treatment_rate) / treatment_n
    )
    absolute_lift = treatment_rate - control_rate
    ci_lower = absolute_lift - z_critical * diff_se
    ci_upper = absolute_lift + z_critical * diff_se

    relative_lift_pct = (absolute_lift / control_rate) * 100.0 if control_rate != 0.0 else 0.0

    alpha = 1.0 - confidence_level
    significant = bool(p_value < alpha)

    return ZTestResult(
        metric=metric,
        control_rate=float(control_rate),
        treatment_rate=float(treatment_rate),
        absolute_lift=float(absolute_lift),
        relative_lift_pct=float(relative_lift_pct),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_stat=float(z_stat),
        p_value=float(p_value),
        significant=significant,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# 3. Mann-Whitney U test
# ---------------------------------------------------------------------------


def mann_whitney_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    metric: str = "sum_gamerounds",
    confidence_level: float = 0.95,
) -> MannWhitneyResult:
    """Run a two-sided Mann-Whitney U test for continuous / count metrics.

    Parameters
    ----------
    control_values:
        Array of per-user metric values for the control group.
    treatment_values:
        Array of per-user metric values for the treatment group.
    metric:
        Label for the metric being tested.
    confidence_level:
        Confidence level used to define the significance threshold.

    Returns
    -------
    MannWhitneyResult
        Dataclass with medians, effect size (rank-biserial), and significance.
    """
    control_values = np.asarray(control_values, dtype=float)
    treatment_values = np.asarray(treatment_values, dtype=float)

    u_stat, p_value = stats.mannwhitneyu(
        control_values, treatment_values, alternative="two-sided"
    )

    n_c = len(control_values)
    n_t = len(treatment_values)

    # Rank-biserial correlation as effect size
    effect_size_r = 1.0 - (2.0 * u_stat) / (n_c * n_t)

    control_median = float(np.median(control_values))
    treatment_median = float(np.median(treatment_values))
    absolute_diff = treatment_median - control_median
    relative_diff_pct = (
        (absolute_diff / control_median) * 100.0 if control_median != 0.0 else 0.0
    )

    alpha = 1.0 - confidence_level
    significant = bool(p_value < alpha)

    return MannWhitneyResult(
        metric=metric,
        control_median=control_median,
        treatment_median=treatment_median,
        absolute_diff=float(absolute_diff),
        relative_diff_pct=float(relative_diff_pct),
        u_stat=float(u_stat),
        p_value=float(p_value),
        effect_size_r=float(effect_size_r),
        significant=significant,
    )


# ---------------------------------------------------------------------------
# 4. Multiple-testing corrections
# ---------------------------------------------------------------------------


def apply_multiple_corrections(
    p_values: list[float],
    metric_names: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Bonferroni and Holm corrections to a collection of p-values.

    Parameters
    ----------
    p_values:
        Raw p-values from individual hypothesis tests.
    metric_names:
        Names of the metrics corresponding to each p-value.
    alpha:
        Family-wise error rate (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: metric, original_p, bonferroni_p, bonferroni_significant,
        holm_p, holm_significant.
    """
    p_array = np.asarray(p_values, dtype=float)

    # Bonferroni
    bonf_reject, bonf_p, _, _ = multipletests(p_array, alpha=alpha, method="bonferroni")

    # Holm
    holm_reject, holm_p, _, _ = multipletests(p_array, alpha=alpha, method="holm")

    return pd.DataFrame(
        {
            "metric": metric_names,
            "original_p": p_array,
            "bonferroni_p": bonf_p,
            "bonferroni_significant": bonf_reject,
            "holm_p": holm_p,
            "holm_significant": holm_reject,
        }
    )


# ---------------------------------------------------------------------------
# 5. Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_diff(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    stat_func: str = "mean",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
    metric: str = "",
) -> BootstrapResult:
    """Estimate the confidence interval of (treatment − control) via bootstrap.

    Parameters
    ----------
    control_values:
        Array of per-user metric values for the control group.
    treatment_values:
        Array of per-user metric values for the treatment group.
    stat_func:
        Aggregation function to use: ``'mean'`` or ``'median'``.
    n_resamples:
        Number of bootstrap resamples (default 10 000).
    confidence_level:
        Width of the percentile CI (default 0.95).
    seed:
        Random seed for reproducibility.
    metric:
        Label for the metric being estimated.

    Returns
    -------
    BootstrapResult
        Dataclass with the observed difference, CI bounds, and all bootstrap
        differences.

    Raises
    ------
    ValueError
        If ``stat_func`` is not ``'mean'`` or ``'median'``.
    """
    control_values = np.asarray(control_values, dtype=float)
    treatment_values = np.asarray(treatment_values, dtype=float)

    if stat_func == "mean":
        _fn = np.mean
    elif stat_func == "median":
        _fn = np.median
    else:
        raise ValueError(f"stat_func must be 'mean' or 'median', got '{stat_func}'.")

    observed_diff = float(_fn(treatment_values) - _fn(control_values))

    rng = np.random.default_rng(seed)
    n_c = len(control_values)
    n_t = len(treatment_values)

    bootstrap_diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        c_sample = rng.choice(control_values, size=n_c, replace=True)
        t_sample = rng.choice(treatment_values, size=n_t, replace=True)
        bootstrap_diffs[i] = _fn(t_sample) - _fn(c_sample)

    lower_pct = (1.0 - confidence_level) / 2.0 * 100.0
    upper_pct = (1.0 + confidence_level) / 2.0 * 100.0
    ci_lower = float(np.percentile(bootstrap_diffs, lower_pct))
    ci_upper = float(np.percentile(bootstrap_diffs, upper_pct))

    return BootstrapResult(
        metric=metric,
        observed_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_diffs=bootstrap_diffs,
        n_resamples=n_resamples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 6. Bayesian Beta-Binomial
# ---------------------------------------------------------------------------


def bayesian_beta_binomial(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    seed: int = 42,
    metric: str = "retention",
) -> BayesianResult:
    """Perform Bayesian Beta-Binomial inference for two proportions.

    Uses conjugate Beta priors.  Posterior for each group is:
        Beta(prior_alpha + successes, prior_beta + failures)

    Parameters
    ----------
    control_successes:
        Number of successes in the control group.
    control_n:
        Total observations in the control group.
    treatment_successes:
        Number of successes in the treatment group.
    treatment_n:
        Total observations in the treatment group.
    prior_alpha:
        Alpha parameter of the Beta prior (default 1 → uniform).
    prior_beta:
        Beta parameter of the Beta prior (default 1 → uniform).
    n_samples:
        Number of Monte-Carlo samples to draw from each posterior.
    seed:
        Random seed for reproducibility.
    metric:
        Label for the metric being analysed.

    Returns
    -------
    BayesianResult
        Dataclass with posterior parameters, probability that treatment is
        better, and expected losses.
    """
    # Posterior parameters
    c_alpha = prior_alpha + control_successes
    c_beta = prior_beta + (control_n - control_successes)
    t_alpha = prior_alpha + treatment_successes
    t_beta = prior_beta + (treatment_n - treatment_successes)

    rng = np.random.default_rng(seed)
    control_samples = rng.beta(c_alpha, c_beta, size=n_samples)
    treatment_samples = rng.beta(t_alpha, t_beta, size=n_samples)

    prob_treatment_better = float(np.mean(treatment_samples > control_samples))

    # Expected loss
    # Loss if we choose treatment and control turns out to be better
    expected_loss_treatment = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))
    # Loss if we choose control and treatment turns out to be better
    expected_loss_control = float(np.mean(np.maximum(treatment_samples - control_samples, 0.0)))

    return BayesianResult(
        metric=metric,
        control_alpha=float(c_alpha),
        control_beta=float(c_beta),
        treatment_alpha=float(t_alpha),
        treatment_beta=float(t_beta),
        prob_treatment_better=prob_treatment_better,
        expected_loss_treatment=expected_loss_treatment,
        expected_loss_control=expected_loss_control,
        control_posterior_samples=control_samples,
        treatment_posterior_samples=treatment_samples,
    )


# ---------------------------------------------------------------------------
# 7. Power analysis
# ---------------------------------------------------------------------------


def compute_power(
    control_n: int,
    treatment_n: int,
    control_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    metric: str = "retention",
) -> PowerResult:
    """Compute achieved power and minimum detectable effect for an A/B test.

    Parameters
    ----------
    control_n:
        Number of users in the control group.
    treatment_n:
        Number of users in the treatment group.
    control_rate:
        Observed conversion/retention rate for the control group.
    treatment_rate:
        Observed conversion/retention rate for the treatment group.
    alpha:
        Significance level (default 0.05).
    metric:
        Label for the metric being analysed.

    Returns
    -------
    PowerResult
        Dataclass with achieved power, MDE at 80 % power, and required sample
        size for 80 % power.
    """
    total_n = control_n + treatment_n
    p_bar = (control_n * control_rate + treatment_n * treatment_rate) / total_n
    ratio = treatment_n / control_n

    # Cohen's h-like effect size for proportions using pooled p
    denom = np.sqrt(p_bar * (1.0 - p_bar))
    effect_size = (treatment_rate - control_rate) / denom if denom > 0 else 0.0

    power_analysis = NormalIndPower()

    # Achieved power at observed sample sizes
    try:
        achieved_power = float(
            power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=control_n,
                ratio=ratio,
                alpha=alpha,
                alternative="two-sided",
            )
        )
    except Exception:
        achieved_power = float("nan")

    # Effect size required for 80 % power at equal allocation with control_n per group
    try:
        es_for_mde = float(
            power_analysis.solve_power(
                power=0.8,
                nobs1=control_n,
                ratio=1.0,
                alpha=alpha,
                alternative="two-sided",
            )
        )
        # Convert back to absolute rate difference: Δ = es * sqrt(p_bar*(1-p_bar))
        mde_at_80_power = float(es_for_mde * np.sqrt(p_bar * (1.0 - p_bar)))
    except Exception:
        mde_at_80_power = float("nan")

    # Sample size per group needed for 80 % power at observed effect size
    try:
        sample_size_for_80_power = int(
            np.ceil(
                power_analysis.solve_power(
                    effect_size=abs(effect_size),
                    power=0.8,
                    ratio=1.0,
                    alpha=alpha,
                    alternative="two-sided",
                )
            )
        )
    except Exception:
        sample_size_for_80_power = -1

    return PowerResult(
        metric=metric,
        n_control=control_n,
        n_treatment=treatment_n,
        observed_effect_size=float(effect_size),
        achieved_power=achieved_power,
        alpha=alpha,
        mde_at_80_power=mde_at_80_power,
        sample_size_for_80_power=sample_size_for_80_power,
    )


# ---------------------------------------------------------------------------
# 8. Peeking simulation
# ---------------------------------------------------------------------------


def simulate_peeking(
    n_simulations: int = 1000,
    n_per_group: int = 1000,
    true_rate: float = 0.44,
    check_points: Optional[list[int]] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate the inflated false-positive rate caused by repeated peeking.

    Runs A/A simulations (both groups have the same true rate) and checks
    significance at multiple intermediate sample sizes to show how early
    stopping inflates Type-I error.

    Parameters
    ----------
    n_simulations:
        Number of simulated experiments (default 1 000).
    n_per_group:
        Maximum sample size per group (default 1 000).
    true_rate:
        True conversion / retention rate used for both groups (null is true).
    check_points:
        List of sample sizes at which to peek.  Defaults to
        ``[100, 200, …, n_per_group]`` in steps of 100.
    alpha:
        Significance threshold for peeking (default 0.05).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: checkpoint, false_positive_rate.
        ``false_positive_rate`` is the fraction of simulations where a false
        positive was detected *at any peek up to and including* that checkpoint.
    """
    if check_points is None:
        step = max(1, n_per_group // 10)
        check_points = list(range(step, n_per_group + 1, step))
        if check_points[-1] != n_per_group:
            check_points.append(n_per_group)

    check_points = sorted(check_points)

    rng = np.random.default_rng(seed)

    # Pre-generate all data: shape (n_simulations, n_per_group) for each group
    control_data = rng.binomial(1, true_rate, size=(n_simulations, n_per_group))
    treatment_data = rng.binomial(1, true_rate, size=(n_simulations, n_per_group))

    # Track whether each simulation has produced a false positive YET
    ever_significant = np.zeros(n_simulations, dtype=bool)

    records: list[dict] = []

    for cp in check_points:
        # Cumulative sums up to checkpoint
        c_successes = control_data[:, :cp].sum(axis=1)
        t_successes = treatment_data[:, :cp].sum(axis=1)

        for sim_idx in range(n_simulations):
            if ever_significant[sim_idx]:
                continue  # already flagged; no need to re-test

            c_s = int(c_successes[sim_idx])
            t_s = int(t_successes[sim_idx])

            # Guard against degenerate cases
            if c_s == 0 and t_s == 0:
                continue
            if c_s == cp and t_s == cp:
                continue

            try:
                _, p_val = proportions_ztest(
                    count=np.array([t_s, c_s]),
                    nobs=np.array([cp, cp]),
                    alternative="two-sided",
                )
                if p_val < alpha:
                    ever_significant[sim_idx] = True
            except Exception:
                pass

        fp_rate = float(ever_significant.sum()) / n_simulations
        records.append({"checkpoint": cp, "false_positive_rate": fp_rate})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 9. Sample-size calculator
# ---------------------------------------------------------------------------


def sample_size_calculator(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> int:
    """Calculate the required per-group sample size for a two-proportion test.

    Uses Cohen's h effect size via :func:`statsmodels.stats.proportion.proportion_effectsize`.

    Parameters
    ----------
    baseline_rate:
        Expected conversion / retention rate in the control group.
    mde:
        Minimum detectable effect (absolute difference in rates).
    alpha:
        Significance level (default 0.05).
    power:
        Desired statistical power (default 0.80).
    ratio:
        Ratio of treatment to control sample sizes (default 1.0 for equal
        groups).

    Returns
    -------
    int
        Required sample size **per control group**.
    """
    p2 = baseline_rate + mde
    # Clamp to [0, 1]
    p2 = float(np.clip(p2, 0.0, 1.0))

    effect_size = proportion_effectsize(p2, baseline_rate)

    n = NormalIndPower().solve_power(
        effect_size=abs(effect_size),
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative="two-sided",
    )

    return int(np.ceil(n))
