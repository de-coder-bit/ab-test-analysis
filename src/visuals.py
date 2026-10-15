"""
visuals.py
==========
Reusable Plotly chart factory functions for the Cookie Cats A/B test dashboard.

All functions return go.Figure objects with a consistent dark-theme palette:
  - Control  (gate_30) : CONTROL_COLOR  = '#00B4D8'  (teal-blue)
  - Treatment (gate_40) : TREATMENT_COLOR = '#EF476F'  (coral-red)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_dist

# ── Global Palette ──────────────────────────────────────────────────────────
CONTROL_COLOR: str = "#00B4D8"
TREATMENT_COLOR: str = "#EF476F"
BG_COLOR: str = "#0E1117"
PAPER_COLOR: str = "#1E2130"
GRID_COLOR: str = "#2D3250"
TEXT_COLOR: str = "#FAFAFA"
MUTED_COLOR: str = "#8B95A1"

_LAYOUT_DEFAULTS: dict = dict(
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
)

_AXIS_DEFAULTS: dict = dict(
    gridcolor=GRID_COLOR,
    zerolinecolor=GRID_COLOR,
    showgrid=True,
)


def _apply_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    """Apply consistent dark-theme layout to a figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(title=xaxis_title, **_AXIS_DEFAULTS),
        yaxis=dict(title=yaxis_title, **_AXIS_DEFAULTS),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── 1. Retention Bar Chart ──────────────────────────────────────────────────

def plot_retention_bars(
    control_rate: float,
    treatment_rate: float,
    ci_control: tuple[float, float],
    ci_treatment: tuple[float, float],
    metric: str = "retention_1",
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Grouped bar chart comparing retention rates with 95% CI error bars.

    Parameters
    ----------
    control_rate : float
        Retention rate for control group (gate_30).
    treatment_rate : float
        Retention rate for treatment group (gate_40).
    ci_control : tuple[float, float]
        (lower, upper) confidence interval for control.
    ci_treatment : tuple[float, float]
        (lower, upper) confidence interval for treatment.
    metric : str
        Name of the metric being plotted.
    confidence_level : float
        Confidence level used (for axis label).

    Returns
    -------
    go.Figure
        Plotly grouped bar chart.
    """
    labels = ["gate_30 (Control)", "gate_40 (Treatment)"]
    rates = [control_rate, treatment_rate]
    colors = [CONTROL_COLOR, TREATMENT_COLOR]
    ci_lowers = [ci_control[0], ci_treatment[0]]
    ci_uppers = [ci_control[1], ci_treatment[1]]
    error_minus = [r - lo for r, lo in zip(rates, ci_lowers)]
    error_plus = [hi - r for r, hi in zip(rates, ci_uppers)]

    fig = go.Figure()
    for i, (label, rate, color, em, ep) in enumerate(
        zip(labels, rates, colors, error_minus, error_plus)
    ):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[rate],
                marker_color=color,
                error_y=dict(type="data", symmetric=False, array=[ep], arrayminus=[em], color=MUTED_COLOR),
                text=[f"{rate:.3%}"],
                textposition="outside",
                textfont=dict(color=TEXT_COLOR, size=14),
            )
        )

    title = f"{metric} Rate by Group ({int(confidence_level*100)}% CI)"
    _apply_layout(fig, title, "", "Retention Rate")
    fig.update_layout(
        yaxis=dict(tickformat=".1%", **_AXIS_DEFAULTS),
        showlegend=False,
        bargap=0.4,
    )
    return fig


# ── 2. Game Rounds Distribution ─────────────────────────────────────────────

def plot_gamerounds_distribution(
    control: pd.Series,
    treatment: pd.Series,
    log_scale: bool = True,
    cap_value: float | None = None,
) -> go.Figure:
    """
    Overlapping histogram of sum_gamerounds for both groups (log scale).

    Parameters
    ----------
    control : pd.Series
        Game rounds for control group.
    treatment : pd.Series
        Game rounds for treatment group.
    log_scale : bool
        Whether to use log scale on x-axis.
    cap_value : float or None
        If provided, draws a vertical line at the cap value.

    Returns
    -------
    go.Figure
        Plotly histogram figure.
    """
    fig = go.Figure()

    for series, name, color in [
        (control, "gate_30 (Control)", CONTROL_COLOR),
        (treatment, "gate_40 (Treatment)", TREATMENT_COLOR),
    ]:
        fig.add_trace(
            go.Histogram(
                x=series,
                name=name,
                marker_color=color,
                opacity=0.6,
                nbinsx=80,
                histnorm="probability density",
            )
        )

    if cap_value is not None:
        fig.add_vline(
            x=cap_value,
            line_dash="dash",
            line_color="#FFD166",
            annotation_text=f"99.9th pct cap ({cap_value:,.0f})",
            annotation_font_color="#FFD166",
        )

    fig.update_layout(barmode="overlay")
    if log_scale:
        fig.update_xaxes(type="log")

    _apply_layout(
        fig,
        "Distribution of Game Rounds per Player (log scale)",
        "Sum Game Rounds (log scale)" if log_scale else "Sum Game Rounds",
        "Density",
    )
    return fig


# ── 3. Bootstrap Distribution ───────────────────────────────────────────────

def plot_bootstrap_distribution(
    bootstrap_diffs: np.ndarray,
    ci_lower: float,
    ci_upper: float,
    observed_diff: float,
    metric: str = "",
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Histogram of bootstrap resampled differences with CI shading.

    Parameters
    ----------
    bootstrap_diffs : np.ndarray
        Array of bootstrapped statistic differences (treatment - control).
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    observed_diff : float
        The observed difference in the original data.
    metric : str
        Name of the metric.
    confidence_level : float
        Confidence level (for title).

    Returns
    -------
    go.Figure
        Histogram with CI lines.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=bootstrap_diffs,
            nbinsx=100,
            marker_color=TREATMENT_COLOR,
            opacity=0.75,
            name="Bootstrap Differences",
        )
    )

    # CI lines
    for x_val, dash, label, color in [
        (ci_lower, "dash", f"{int(confidence_level*100)}% CI Lower", "#FFD166"),
        (ci_upper, "dash", f"{int(confidence_level*100)}% CI Upper", "#FFD166"),
        (observed_diff, "solid", "Observed Diff", "#06D6A0"),
        (0, "dot", "No Effect (0)", MUTED_COLOR),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash=dash,
            line_color=color,
            line_width=2,
            annotation_text=label,
            annotation_font_color=color,
            annotation_position="top",
        )

    _apply_layout(
        fig,
        f"Bootstrap Distribution of Difference in {metric} (n=10,000 resamples)",
        "Difference (Treatment − Control)",
        "Count",
    )
    return fig


# ── 4. Bayesian Posterior Distributions ────────────────────────────────────

def plot_posterior_distributions(
    control_alpha: float,
    control_beta_param: float,
    treatment_alpha: float,
    treatment_beta_param: float,
    metric: str = "retention",
    prob_treatment_better: float | None = None,
) -> go.Figure:
    """
    Overlapping Beta posterior distribution PDFs for control and treatment.

    Parameters
    ----------
    control_alpha : float
        Alpha parameter of control posterior Beta distribution.
    control_beta_param : float
        Beta parameter of control posterior Beta distribution.
    treatment_alpha : float
        Alpha parameter of treatment posterior Beta distribution.
    treatment_beta_param : float
        Beta parameter of treatment posterior Beta distribution.
    metric : str
        Name of the metric.
    prob_treatment_better : float or None
        P(treatment > control) for annotation.

    Returns
    -------
    go.Figure
        PDF plot of both posteriors.
    """
    x = np.linspace(0.05, 0.95, 1000)

    control_pdf = beta_dist.pdf(x, control_alpha, control_beta_param)
    treatment_pdf = beta_dist.pdf(x, treatment_alpha, treatment_beta_param)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=control_pdf,
            name="gate_30 (Control)",
            mode="lines",
            line=dict(color=CONTROL_COLOR, width=3),
            fill="tozeroy",
            fillcolor=f"rgba({int(CONTROL_COLOR[1:3], 16)},{int(CONTROL_COLOR[3:5], 16)},{int(CONTROL_COLOR[5:], 16)},0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=treatment_pdf,
            name="gate_40 (Treatment)",
            mode="lines",
            line=dict(color=TREATMENT_COLOR, width=3),
            fill="tozeroy",
            fillcolor=f"rgba({int(TREATMENT_COLOR[1:3], 16)},{int(TREATMENT_COLOR[3:5], 16)},{int(TREATMENT_COLOR[5:], 16)},0.15)",
        )
    )

    title = f"Bayesian Posterior Distributions — {metric}"
    if prob_treatment_better is not None:
        title += f"<br><sup>P(treatment > control) = {prob_treatment_better:.3%}</sup>"

    _apply_layout(fig, title, "Retention Rate", "Probability Density")
    fig.update_xaxes(tickformat=".1%")
    return fig


# ── 5. Peeking Simulation ───────────────────────────────────────────────────

def plot_peeking_simulation(peeking_df: pd.DataFrame, alpha: float = 0.05) -> go.Figure:
    """
    Line chart showing cumulative false positive rate vs. number of peeks.

    Parameters
    ----------
    peeking_df : pd.DataFrame
        DataFrame with columns 'checkpoint' and 'false_positive_rate'.
    alpha : float
        Nominal alpha level (shown as horizontal reference line).

    Returns
    -------
    go.Figure
        Line plot of false positive inflation.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=peeking_df["checkpoint"],
            y=peeking_df["false_positive_rate"],
            mode="lines+markers",
            name="Observed False Positive Rate",
            line=dict(color=TREATMENT_COLOR, width=3),
            marker=dict(size=6),
        )
    )

    fig.add_hline(
        y=alpha,
        line_dash="dash",
        line_color="#FFD166",
        line_width=2,
        annotation_text=f"Nominal α = {alpha}",
        annotation_font_color="#FFD166",
    )

    _apply_layout(
        fig,
        "⚠️ Peeking Problem: False Positive Rate Inflation",
        "Sample Size at Peek",
        "Cumulative False Positive Rate",
    )
    fig.update_yaxes(tickformat=".1%")
    return fig


# ── 6. Power Curve ──────────────────────────────────────────────────────────

def plot_power_curve(
    baseline_rate: float,
    mde_values: list[float],
    sample_sizes: list[int],
    alpha: float = 0.05,
    current_n: int | None = None,
) -> go.Figure:
    """
    Heatmap-style power curve: power vs. sample size for various MDEs.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion/retention rate.
    mde_values : list[float]
        List of minimum detectable effects (absolute, e.g. 0.01 = 1pp).
    sample_sizes : list[int]
        List of sample sizes per group.
    alpha : float
        Significance level.
    current_n : int or None
        Current experiment sample size to annotate on the plot.

    Returns
    -------
    go.Figure
        Power curve figure.
    """
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    analysis = NormalIndPower()
    fig = go.Figure()

    colors_power = px.colors.sequential.Viridis
    color_step = max(1, len(colors_power) // max(len(mde_values), 1))

    for i, mde in enumerate(mde_values):
        p2 = baseline_rate + mde
        if p2 >= 1.0 or p2 <= 0.0:
            continue
        es = proportion_effectsize(p2, baseline_rate)
        powers = []
        for n in sample_sizes:
            try:
                pwr = analysis.solve_power(effect_size=abs(es), nobs1=n, ratio=1.0, alpha=alpha, alternative="two-sided")
            except Exception:
                pwr = np.nan
            powers.append(min(pwr, 1.0) if not np.isnan(pwr) else np.nan)

        color_idx = min(i * color_step, len(colors_power) - 1)
        fig.add_trace(
            go.Scatter(
                x=sample_sizes,
                y=powers,
                mode="lines",
                name=f"MDE = {mde:.1%}",
                line=dict(color=colors_power[color_idx], width=2),
            )
        )

    fig.add_hline(y=0.80, line_dash="dash", line_color="#06D6A0",
                  annotation_text="80% Power", annotation_font_color="#06D6A0")
    fig.add_hline(y=0.90, line_dash="dot", line_color="#FFD166",
                  annotation_text="90% Power", annotation_font_color="#FFD166")

    if current_n is not None:
        fig.add_vline(x=current_n, line_dash="dash", line_color=MUTED_COLOR,
                      annotation_text=f"Current n={current_n:,}", annotation_font_color=MUTED_COLOR)

    _apply_layout(fig, "Statistical Power vs. Sample Size", "Sample Size per Group", "Statistical Power")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return fig


# ── 7. Segment Comparison Chart ─────────────────────────────────────────────

def plot_segment_comparison(segment_df: pd.DataFrame, metric: str = "retention_7") -> go.Figure:
    """
    Grouped bar chart comparing metric by engagement segment and version.

    Parameters
    ----------
    segment_df : pd.DataFrame
        DataFrame with columns: segment, gate_30_rate, gate_40_rate, gate_30_n, gate_40_n.
    metric : str
        Metric name for title.

    Returns
    -------
    go.Figure
        Grouped bar chart.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="gate_30 (Control)",
            x=segment_df["segment"],
            y=segment_df["gate_30_rate"],
            marker_color=CONTROL_COLOR,
            text=[f"{r:.1%}" for r in segment_df["gate_30_rate"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="gate_40 (Treatment)",
            x=segment_df["segment"],
            y=segment_df["gate_40_rate"],
            marker_color=TREATMENT_COLOR,
            text=[f"{r:.1%}" for r in segment_df["gate_40_rate"]],
            textposition="outside",
        )
    )

    _apply_layout(
        fig,
        f"{metric} by Engagement Segment",
        "Engagement Segment (Game Rounds)",
        f"{metric} Rate",
    )
    fig.update_layout(barmode="group", yaxis=dict(tickformat=".1%", **_AXIS_DEFAULTS))
    return fig


# ── 8. Retention Over Time (EDA placeholder) ─────────────────────────────

def plot_version_composition(version_counts: pd.Series) -> go.Figure:
    """
    Pie chart showing the split between control and treatment groups.

    Parameters
    ----------
    version_counts : pd.Series
        Series with index ['gate_30', 'gate_40'] and values as counts.

    Returns
    -------
    go.Figure
        Donut pie chart.
    """
    fig = go.Figure(
        go.Pie(
            labels=["gate_30 (Control)", "gate_40 (Treatment)"],
            values=[version_counts.get("gate_30", 0), version_counts.get("gate_40", 0)],
            hole=0.5,
            marker=dict(colors=[CONTROL_COLOR, TREATMENT_COLOR]),
            textinfo="label+percent+value",
            textfont=dict(color=TEXT_COLOR),
        )
    )
    fig.update_layout(
        title=dict(text="Group Split", font=dict(size=16, color=TEXT_COLOR)),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── 9. Zero-Round Players Bar ────────────────────────────────────────────────

def plot_zero_round_comparison(
    control_zero_pct: float,
    treatment_zero_pct: float,
) -> go.Figure:
    """
    Bar chart comparing percentage of zero-round players across groups.

    Parameters
    ----------
    control_zero_pct : float
        Fraction of control players with 0 game rounds.
    treatment_zero_pct : float
        Fraction of treatment players with 0 game rounds.

    Returns
    -------
    go.Figure
        Simple bar chart.
    """
    fig = go.Figure(
        go.Bar(
            x=["gate_30 (Control)", "gate_40 (Treatment)"],
            y=[control_zero_pct, treatment_zero_pct],
            marker_color=[CONTROL_COLOR, TREATMENT_COLOR],
            text=[f"{control_zero_pct:.2%}", f"{treatment_zero_pct:.2%}"],
            textposition="outside",
        )
    )
    _apply_layout(fig, "Players with Zero Game Rounds", "Group", "Fraction of Players")
    fig.update_layout(
        yaxis=dict(tickformat=".1%", **_AXIS_DEFAULTS),
        showlegend=False,
    )
    return fig


# ── 10. KPI Metric Cards (returns dict for st.metric) ────────────────────────

def build_kpi_data(
    z_result_r1,
    z_result_r7,
    mw_result,
    srm_result,
) -> list[dict]:
    """
    Build a list of KPI card data dictionaries from analysis results.

    Parameters
    ----------
    z_result_r1 : ZTestResult
        Z-test result for retention_1.
    z_result_r7 : ZTestResult
        Z-test result for retention_7.
    mw_result : MannWhitneyResult
        Mann-Whitney result for sum_gamerounds.
    srm_result : SRMResult
        SRM check result.

    Returns
    -------
    list[dict]
        Each dict has keys: label, value, delta, delta_color, help_text.
    """
    return [
        {
            "label": "1-Day Retention (Control)",
            "value": f"{z_result_r1.control_rate:.2%}",
            "delta": f"{z_result_r1.absolute_lift:+.2%} ({z_result_r1.relative_lift_pct:+.1f}% rel.)",
            "delta_color": "normal" if z_result_r1.absolute_lift >= 0 else "inverse",
            "help_text": f"p = {z_result_r1.p_value:.4f} | {'✅ Significant' if z_result_r1.significant else '❌ Not significant'}",
        },
        {
            "label": "7-Day Retention (Control)",
            "value": f"{z_result_r7.control_rate:.2%}",
            "delta": f"{z_result_r7.absolute_lift:+.2%} ({z_result_r7.relative_lift_pct:+.1f}% rel.)",
            "delta_color": "normal" if z_result_r7.absolute_lift >= 0 else "inverse",
            "help_text": f"p = {z_result_r7.p_value:.4f} | {'✅ Significant' if z_result_r7.significant else '❌ Not significant'}",
        },
        {
            "label": "Median Game Rounds (Control)",
            "value": f"{mw_result.control_median:.0f}",
            "delta": f"{mw_result.absolute_diff:+.1f} ({mw_result.relative_diff_pct:+.1f}%)",
            "delta_color": "normal" if mw_result.absolute_diff >= 0 else "inverse",
            "help_text": f"p = {mw_result.p_value:.4f} | {'✅ Significant' if mw_result.significant else '❌ Not significant'}",
        },
        {
            "label": "SRM Check",
            "value": "✅ PASSED" if srm_result.passed else "⚠️ FAILED",
            "delta": f"p = {srm_result.p_value:.4f}",
            "delta_color": "normal" if srm_result.passed else "inverse",
            "help_text": srm_result.message,
        },
    ]
