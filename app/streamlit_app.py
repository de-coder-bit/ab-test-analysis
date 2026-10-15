"""
Cookie Cats A/B Test Analysis — Streamlit Dashboard
=====================================================
Run from project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── make src/ importable when running from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_and_clean, split_by_version
from src.stats_tests import (
    SRMResult,
    ZTestResult,
    MannWhitneyResult,
    BootstrapResult,
    BayesianResult,
    PowerResult,
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
from src.visuals import (
    CONTROL_COLOR,
    TREATMENT_COLOR,
    build_kpi_data,
    plot_bootstrap_distribution,
    plot_gamerounds_distribution,
    plot_posterior_distributions,
    plot_power_curve,
    plot_retention_bars,
    plot_segment_comparison,
    plot_version_composition,
    plot_zero_round_comparison,
    plot_peeking_simulation,
)

# ── PAGE CONFIG — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Cookie Cats A/B Test Analysis",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(remove_outliers: bool) -> tuple:
    """Load and clean the Cookie Cats dataset.

    Parameters
    ----------
    remove_outliers:
        When True, cap extreme game-round values; otherwise keep all rows.

    Returns
    -------
    tuple
        ``(df, report_dict)`` as produced by :func:`load_and_clean`.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cookie_cats.csv"
    return load_and_clean(data_path, remove_outliers=remove_outliers)


@st.cache_data
def run_all_analyses(remove_outliers: bool, confidence_level: float) -> dict:
    """Run every statistical analysis and return results in a single dict.

    Parameters
    ----------
    remove_outliers:
        Passed through to :func:`load_data`.
    confidence_level:
        E.g. 0.95 → alpha = 0.05.

    Returns
    -------
    dict
        All computed results, raw data frames, and the cleaning report.
    """
    df, report = load_data(remove_outliers)
    control_df, treatment_df = split_by_version(df)
    alpha = 1.0 - confidence_level

    results: dict = {}

    # ── SRM ──────────────────────────────────────────────────────────────────
    results["srm"] = check_srm(report["control_size"], report["treatment_size"])

    # ── Z-tests (proportions) ────────────────────────────────────────────────
    for metric in ["retention_1", "retention_7"]:
        c_succ = int(control_df[metric].sum())
        c_n = len(control_df)
        t_succ = int(treatment_df[metric].sum())
        t_n = len(treatment_df)
        results[f"ztest_{metric}"] = two_prop_z_test(
            c_succ, c_n, t_succ, t_n,
            metric=metric,
            confidence_level=confidence_level,
        )

    # ── Mann-Whitney U (game rounds) ─────────────────────────────────────────
    results["mw"] = mann_whitney_test(
        control_df["sum_gamerounds"].values,
        treatment_df["sum_gamerounds"].values,
        confidence_level=confidence_level,
    )

    # ── Multiple-testing corrections ─────────────────────────────────────────
    p_vals = [
        results["ztest_retention_1"].p_value,
        results["ztest_retention_7"].p_value,
        results["mw"].p_value,
    ]
    results["corrections"] = apply_multiple_corrections(
        p_vals,
        ["retention_1", "retention_7", "sum_gamerounds"],
        alpha=alpha,
    )

    # ── Bootstrap: retention_7 (mean) ────────────────────────────────────────
    c_r7 = control_df["retention_7"].values.astype(float)
    t_r7 = treatment_df["retention_7"].values.astype(float)
    results["bootstrap_r7"] = bootstrap_diff(
        c_r7, t_r7,
        stat_func="mean",
        n_resamples=10_000,
        confidence_level=confidence_level,
        seed=42,
        metric="retention_7",
    )

    # ── Bootstrap: game rounds (median) ──────────────────────────────────────
    results["bootstrap_gr"] = bootstrap_diff(
        control_df["sum_gamerounds"].values.astype(float),
        treatment_df["sum_gamerounds"].values.astype(float),
        stat_func="median",
        n_resamples=10_000,
        confidence_level=confidence_level,
        seed=42,
        metric="sum_gamerounds",
    )

    # ── Bayesian Beta-Binomial ────────────────────────────────────────────────
    for metric in ["retention_1", "retention_7"]:
        c_succ = int(control_df[metric].sum())
        c_n = len(control_df)
        t_succ = int(treatment_df[metric].sum())
        t_n = len(treatment_df)
        results[f"bayes_{metric}"] = bayesian_beta_binomial(
            c_succ, c_n, t_succ, t_n, seed=42, metric=metric
        )

    # ── Power analysis ────────────────────────────────────────────────────────
    for metric in ["retention_1", "retention_7"]:
        zr: ZTestResult = results[f"ztest_{metric}"]
        results[f"power_{metric}"] = compute_power(
            report["control_size"],
            report["treatment_size"],
            zr.control_rate,
            zr.treatment_rate,
            alpha=alpha,
            metric=metric,
        )

    # ── Peeking simulation ────────────────────────────────────────────────────
    results["peeking"] = simulate_peeking(
        n_simulations=1_000,
        n_per_group=1_000,
        true_rate=0.44,
        check_points=list(range(50, 1_050, 50)),
        alpha=alpha,
        seed=42,
    )

    # ── Segmentation by engagement level ─────────────────────────────────────
    def _get_segment(rounds: int) -> str:
        if rounds == 0:
            return "0 rounds"
        elif rounds <= 20:
            return "1-20 rounds"
        elif rounds <= 100:
            return "21-100 rounds"
        return "100+ rounds"

    df_seg = df.copy()
    df_seg["segment"] = df_seg["sum_gamerounds"].apply(_get_segment)
    seg_order = ["0 rounds", "1-20 rounds", "21-100 rounds", "100+ rounds"]

    seg_rows = []
    for seg in seg_order:
        seg_df = df_seg[df_seg["segment"] == seg]
        c_seg = seg_df[seg_df["version"] == "gate_30"]
        t_seg = seg_df[seg_df["version"] == "gate_40"]
        for m in ["retention_1", "retention_7"]:
            seg_rows.append(
                {
                    "segment": seg,
                    "metric": m,
                    "gate_30_rate": c_seg[m].mean() if len(c_seg) > 0 else 0.0,
                    "gate_40_rate": t_seg[m].mean() if len(t_seg) > 0 else 0.0,
                    "gate_30_n": len(c_seg),
                    "gate_40_n": len(t_seg),
                }
            )
    results["segments"] = pd.DataFrame(seg_rows)

    results["df"] = df
    results["control_df"] = control_df
    results["treatment_df"] = treatment_df
    results["report"] = report
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Verdict helper
# ─────────────────────────────────────────────────────────────────────────────

def get_verdict(results: dict, confidence_level: float) -> tuple[str, str, str]:
    """Derive a plain-English shipping recommendation.

    Parameters
    ----------
    results:
        The dict produced by :func:`run_all_analyses`.
    confidence_level:
        Chosen confidence level (e.g. 0.95).

    Returns
    -------
    tuple
        ``(verdict_label, hex_color, reasoning_text)``
    """
    zr7: ZTestResult = results["ztest_retention_7"]
    alpha = 1.0 - confidence_level
    is_sig = zr7.p_value < alpha

    lift = getattr(zr7, "absolute_lift", None)
    if lift is None:
        # fall back to computing from rates
        lift = zr7.treatment_rate - zr7.control_rate

    if is_sig and lift < 0:
        verdict = "🚫 DON'T SHIP gate_40"
        color = "#EF476F"
        reasoning = (
            f"The 7-day retention rate is **statistically significantly lower** for gate_40 "
            f"(p = {zr7.p_value:.4f} < α = {alpha:.2f}). "
            f"The absolute lift is **{lift*100:.2f} pp**, meaning gate_40 costs us retained players. "
            f"Moving the gate from level 30 to level 40 harms long-term engagement. "
            f"Recommend keeping gate at level 30."
        )
    elif is_sig and lift > 0:
        verdict = "✅ SHIP gate_40"
        color = "#06D6A0"
        reasoning = (
            f"The 7-day retention rate is **statistically significantly higher** for gate_40 "
            f"(p = {zr7.p_value:.4f} < α = {alpha:.2f}). "
            f"The absolute lift is **+{lift*100:.2f} pp**, meaning gate_40 retains more players. "
            f"Recommend shipping gate_40."
        )
    else:
        verdict = "⏳ NEED MORE DATA"
        color = "#FFD166"
        reasoning = (
            f"The 7-day retention difference did **not** reach statistical significance "
            f"(p = {zr7.p_value:.4f}, α = {alpha:.2f}). "
            f"We cannot confidently attribute the observed difference to the gate change. "
            f"Consider running the experiment longer or increasing the sample size."
        )

    return verdict, color, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🍪 Cookie Cats A/B Test")
    st.divider()

    primary_metric = st.selectbox(
        "Primary Metric",
        ["retention_7", "retention_1", "sum_gamerounds"],
        index=0,
    )

    confidence_level: float = st.select_slider(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    remove_outliers: bool = st.toggle("Remove Outliers", value=False)
    st.divider()

    # Load data for sidebar stats — handle missing file
    try:
        _df_sidebar, _report_sidebar = load_data(remove_outliers)
        st.markdown("**Group Sizes**")
        col_a, col_b = st.columns(2)
        col_a.metric("gate_30 (Control)", f"{_report_sidebar['control_size']:,}")
        col_b.metric("gate_40 (Treat.)", f"{_report_sidebar['treatment_size']:,}")
        st.caption(f"Total users: {_report_sidebar['n_after']:,}")
    except FileNotFoundError:
        st.warning("Data file not found.")

    st.divider()
    st.info("💡 Gate_30 = Control | Gate_40 = Treatment")

# ─────────────────────────────────────────────────────────────────────────────
# Load data & run analyses — with graceful error handling
# ─────────────────────────────────────────────────────────────────────────────

try:
    results = run_all_analyses(remove_outliers, confidence_level)
except FileNotFoundError:
    st.error(
        "**Data file not found.** Please place `cookie_cats.csv` in the "
        "`ab-test-analysis/data/` directory and reload this page."
    )
    st.stop()

report = results["report"]
df = results["df"]
control_df = results["control_df"]
treatment_df = results["treatment_df"]
alpha = 1.0 - confidence_level

verdict, verdict_color, verdict_reasoning = get_verdict(results, confidence_level)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

(
    tab1, tab2, tab3, tab4,
    tab5, tab6, tab7, tab8, tab9,
) = st.tabs(
    [
        "📊 Executive Summary",
        "🔍 Data Quality & SRM",
        "📈 Frequentist Results",
        "🎲 Bayesian Results",
        "🔄 Bootstrap",
        "⚡ Power & Sample Size",
        "👀 Peeking Simulation",
        "🎯 Segments",
        "💰 Business Impact",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Executive Summary")

    # ── Verdict card ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background-color: {verdict_color}22;
            border-left: 6px solid {verdict_color};
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 24px;
        ">
            <h2 style="color: {verdict_color}; margin: 0 0 8px 0;">{verdict}</h2>
            <p style="margin: 0; font-size: 1rem;">{verdict_reasoning}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4 KPI columns ─────────────────────────────────────────────────────────
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    zr1: ZTestResult = results["ztest_retention_1"]
    zr7: ZTestResult = results["ztest_retention_7"]
    mw: MannWhitneyResult = results["mw"]
    srm: SRMResult = results["srm"]

    lift_r1 = getattr(zr1, "absolute_lift", zr1.treatment_rate - zr1.control_rate)
    lift_r7 = getattr(zr7, "absolute_lift", zr7.treatment_rate - zr7.control_rate)

    kpi_col1.metric(
        label="1-Day Retention Lift",
        value=f"{lift_r1*100:+.2f} pp",
        delta=f"p = {zr1.p_value:.4f}",
        delta_color="inverse" if lift_r1 < 0 else "normal",
    )
    kpi_col2.metric(
        label="7-Day Retention Lift",
        value=f"{lift_r7*100:+.2f} pp",
        delta=f"p = {zr7.p_value:.4f}",
        delta_color="inverse" if lift_r7 < 0 else "normal",
    )

    median_ctrl = float(np.median(control_df["sum_gamerounds"].values))
    median_trt = float(np.median(treatment_df["sum_gamerounds"].values))
    rounds_diff = median_trt - median_ctrl
    kpi_col3.metric(
        label="Game Rounds (median diff)",
        value=f"{rounds_diff:+.1f}",
        delta=f"p = {mw.p_value:.4f}",
        delta_color="inverse" if rounds_diff < 0 else "normal",
    )

    srm_status = "✅ PASS" if srm.passed else "❌ FAIL"
    kpi_col4.metric(
        label="SRM Check",
        value=srm_status,
        delta=f"p = {srm.p_value:.4f}",
        delta_color="off",
    )

    st.divider()

    # ── Plain English summary ─────────────────────────────────────────────────
    st.subheader("Plain-English Summary")
    st.markdown(
        f"""
        We ran an A/B test on **Cookie Cats**, a mobile puzzle game, comparing two gate
        placements: **gate_30** (control, {report['control_size']:,} players) vs
        **gate_40** (treatment, {report['treatment_size']:,} players).

        - **1-day retention** (did the player come back the next day?):
          gate_30 = **{zr1.control_rate*100:.1f}%**, gate_40 = **{zr1.treatment_rate*100:.1f}%**
          (difference: **{lift_r1*100:+.2f} pp**, p = {zr1.p_value:.4f}).
        - **7-day retention** (did the player still play after a week?):
          gate_30 = **{zr7.control_rate*100:.1f}%**, gate_40 = **{zr7.treatment_rate*100:.1f}%**
          (difference: **{lift_r7*100:+.2f} pp**, p = {zr7.p_value:.4f}).
        - The **sample ratio mismatch (SRM) check {srm_status}** — meaning the split
          between groups {"looks as expected." if srm.passed else "is suspicious and should be investigated before trusting results."}

        Overall recommendation: **{verdict}**
        """
    )

    st.divider()

    # ── Download final report ─────────────────────────────────────────────────
    report_path = Path(__file__).resolve().parents[1] / "reports" / "final_report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as fh:
            report_text = fh.read()
        st.download_button(
            label="📥 Download Full Analysis Report (Markdown)",
            data=report_text,
            file_name="cookie_cats_final_report.md",
            mime="text/markdown",
        )
    else:
        st.info("Full report not yet generated. Run the analysis pipeline first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Quality & SRM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Data Quality & SRM")

    # ── Data quality summary table ────────────────────────────────────────────
    st.subheader("Data Quality Report")
    quality_df = pd.DataFrame(
        {
            "Check": [
                "Rows before cleaning",
                "Rows after cleaning",
                "Duplicates removed",
                "Missing values",
                "Outlier action",
                "Outlier cap value",
                "Rows affected by outlier cap",
                "Zero-round players",
                "gate_30 users",
                "gate_40 users",
            ],
            "Value": [
                f"{report['n_before']:,}",
                f"{report['n_after']:,}",
                f"{report['n_duplicates_removed']:,}",
                str(report.get("missing_values", 0)),
                report.get("outlier_action", "none"),
                str(report.get("outlier_cap_value", "N/A")),
                f"{report.get('n_outliers_affected', 0):,}",
                f"{report.get('zero_round_players', 0):,}",
                f"{report['control_size']:,}",
                f"{report['treatment_size']:,}",
            ],
        }
    )
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Game-rounds distribution ──────────────────────────────────────────────
    st.subheader("Game Rounds Distribution")
    with st.expander("ℹ️ What is this chart showing?", expanded=False):
        st.markdown(
            "This histogram shows how many game rounds players played. "
            "Notice the **very long right tail** — a small number of players play "
            "thousands of rounds. These are the outliers that can distort mean-based "
            "statistics."
        )
    fig_dist = plot_gamerounds_distribution(df)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── Version composition ───────────────────────────────────────────────────
    st.subheader("Version Composition")
    fig_pie = plot_version_composition(df)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── SRM ───────────────────────────────────────────────────────────────────
    st.subheader("Sample Ratio Mismatch (SRM) Check")

    srm_badge_color = "#06D6A0" if srm.passed else "#EF476F"
    srm_label = "✅ PASS — No mismatch detected" if srm.passed else "❌ FAIL — Mismatch detected"
    st.markdown(
        f"""
        <div style="
            background: {srm_badge_color}22;
            border-left: 6px solid {srm_badge_color};
            border-radius: 8px;
            padding: 16px 20px;
        ">
            <h3 style="color: {srm_badge_color}; margin:0 0 6px 0;">{srm_label}</h3>
            <b>χ² statistic:</b> {srm.chi2_stat:.4f} &nbsp;|&nbsp;
            <b>p-value:</b> {srm.p_value:.4f} &nbsp;|&nbsp;
            <b>Expected ratio:</b> 50 / 50
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **What is SRM?** In an A/B test we expect both groups to be roughly equal in size
        (since users are randomly assigned). If the split is significantly different from
        50/50, it suggests something went wrong during assignment — a bug in the
        randomisation logic, logging issues, or selection bias. A failed SRM check means
        we cannot trust the experiment results.
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Frequentist Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📈 Frequentist Results")

    st.markdown(
        "A **frequentist p-value** answers: *if there were truly no difference between "
        "gate_30 and gate_40, how often would we see a result this extreme purely by chance?* "
        f"We use α = {alpha:.2f} as our significance threshold."
    )

    # ── Retention side-by-side ────────────────────────────────────────────────
    col_r1, col_r7 = st.columns(2)

    for col, metric, zr in [
        (col_r1, "retention_1", results["ztest_retention_1"]),
        (col_r7, "retention_7", results["ztest_retention_7"]),
    ]:
        label = "1-Day Retention" if metric == "retention_1" else "7-Day Retention"
        lift = getattr(zr, "absolute_lift", zr.treatment_rate - zr.control_rate)
        rel_lift = lift / zr.control_rate if zr.control_rate else 0.0
        sig_flag = "✅ Significant" if zr.p_value < alpha else "⚠️ Not Significant"

        with col:
            st.subheader(label)
            fig_bar = plot_retention_bars(
                control_df, treatment_df, metric=metric
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Control rate", f"{zr.control_rate*100:.2f}%")
            m2.metric("Treatment rate", f"{zr.treatment_rate*100:.2f}%")

            m3, m4 = st.columns(2)
            m3.metric("Absolute lift", f"{lift*100:+.2f} pp")
            m4.metric("Relative lift", f"{rel_lift*100:+.1f}%")

            with st.expander("🔬 Technical Details", expanded=False):
                st.markdown(
                    f"| Stat | Value |\n"
                    f"|---|---|\n"
                    f"| Z-statistic | `{zr.z_stat:.4f}` |\n"
                    f"| p-value | `{zr.p_value:.6f}` |\n"
                    f"| {int(confidence_level*100)}% CI (absolute lift) | "
                    f"`[{zr.ci_lower*100:.3f} pp, {zr.ci_upper*100:.3f} pp]` |\n"
                    f"| Significance | {sig_flag} |"
                )

    st.divider()

    # ── Mann-Whitney U ────────────────────────────────────────────────────────
    st.subheader("Mann-Whitney U Test — Game Rounds")
    st.markdown(
        "Because game rounds are heavily skewed (most players play very few rounds, "
        "a few play thousands), we use a **non-parametric test** that doesn't assume "
        "a normal distribution. It compares whether one group tends to play *more* rounds."
    )

    mw_sig = "✅ Significant" if mw.p_value < alpha else "⚠️ Not Significant"
    mw_cols = st.columns(4)
    mw_cols[0].metric("U statistic", f"{mw.u_stat:,.0f}")
    mw_cols[1].metric("p-value", f"{mw.p_value:.4f}")
    mw_cols[2].metric("Effect size (r)", f"{getattr(mw, 'effect_size', 0):.4f}")
    mw_cols[3].metric("Significance", mw_sig)

    with st.expander("🔬 Effect size interpretation", expanded=False):
        effect = getattr(mw, "effect_size", 0)
        if abs(effect) < 0.1:
            interpretation = "negligible"
        elif abs(effect) < 0.3:
            interpretation = "small"
        elif abs(effect) < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        st.markdown(
            f"The rank-biserial correlation r = {effect:.4f} represents a **{interpretation}** "
            f"practical effect. Even if statistically significant, a negligible effect "
            f"size means the real-world difference is minimal."
        )

    st.divider()

    # ── Multiple corrections ──────────────────────────────────────────────────
    st.subheader("Multiple Testing Corrections")
    st.markdown(
        "When we run **multiple tests** on the same data, the chance of a false positive "
        "grows. For example, with 3 tests at α = 5%, there's a ~14% chance of at least "
        "one false alarm. Corrections like **Bonferroni** (conservative) and "
        "**Benjamini-Hochberg** (less conservative) adjust our thresholds."
    )

    corrections_df: pd.DataFrame = results["corrections"]
    st.dataframe(corrections_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Bayesian Results
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🎲 Bayesian Results")

    st.markdown(
        "Unlike frequentist p-values, **Bayesian analysis** gives us direct probability "
        "statements: *'There is a 92% probability that gate_40 is worse than gate_30.'* "
        "We use a **Beta-Binomial** model with an uninformative prior (Beta(1,1)) so the "
        "data speaks for itself."
    )

    bay_tab1, bay_tab2 = st.tabs(["1-Day Retention", "7-Day Retention"])

    for bay_tab, metric in [(bay_tab1, "retention_1"), (bay_tab2, "retention_7")]:
        bayes: BayesianResult = results[f"bayes_{metric}"]
        label = "1-Day" if metric == "retention_1" else "7-Day"

        with bay_tab:
            fig_post = plot_posterior_distributions(
                results[f"bayes_{metric}"], metric=metric
            )
            st.plotly_chart(fig_post, use_container_width=True)

            b1, b2, b3 = st.columns(3)
            prob_better = getattr(bayes, "prob_treatment_better", None)
            exp_loss = getattr(bayes, "expected_loss", None)
            rope = getattr(bayes, "rope_probability", None)

            b1.metric(
                "P(gate_40 > gate_30)",
                f"{prob_better*100:.1f}%" if prob_better is not None else "N/A",
            )
            b2.metric(
                "Expected Loss (if we ship gate_40)",
                f"{exp_loss*100:.3f} pp" if exp_loss is not None else "N/A",
            )
            b3.metric(
                "P(practically equivalent)",
                f"{rope*100:.1f}%" if rope is not None else "N/A",
            )

            decide = "better" if (prob_better or 0) > 0.5 else "worse"
            confidence_str = f"{(prob_better or 0.5)*100:.0f}%" if (prob_better or 0.5) > 0.5 else f"{(1 - (prob_better or 0.5))*100:.0f}%"
            st.info(
                f"**Interpretation:** The data suggests a **{confidence_str} probability** "
                f"that gate_40 is **{decide}** than gate_30 on {label} retention. "
                f"The expected loss quantifies how much retention we'd give up if we ship "
                f"gate_40 and it turns out to be worse."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Bootstrap
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔄 Bootstrap Analysis")

    st.markdown(
        "**Bootstrap** is a simulation technique that re-samples the data thousands of "
        "times to estimate the uncertainty in our statistics. Think of it as repeatedly "
        "drawing a new random sample from the data we have, computing the difference "
        "each time, and looking at the distribution of those differences. "
        "It makes no assumptions about the data distribution."
    )

    boot_tab1, boot_tab2 = st.tabs(
        ["7-Day Retention (mean)", "Game Rounds (median)"]
    )

    for boot_tab, key, label, metric in [
        (boot_tab1, "bootstrap_r7", "7-Day Retention", "retention_7"),
        (boot_tab2, "bootstrap_gr", "Game Rounds", "sum_gamerounds"),
    ]:
        bs: BootstrapResult = results[key]
        with boot_tab:
            fig_bs = plot_bootstrap_distribution(bs, metric=metric)
            st.plotly_chart(fig_bs, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Observed difference", f"{bs.observed_diff:.5f}")
            c2.metric(
                f"{int(confidence_level*100)}% CI — Lower",
                f"{bs.ci_lower:.5f}",
            )
            c3.metric(
                f"{int(confidence_level*100)}% CI — Upper",
                f"{bs.ci_upper:.5f}",
            )

            includes_zero = bs.ci_lower < 0 < bs.ci_upper
            if includes_zero:
                st.warning(
                    f"The bootstrap {int(confidence_level*100)}% confidence interval "
                    f"**includes zero** ({bs.ci_lower:.4f} to {bs.ci_upper:.4f}), "
                    f"meaning we cannot rule out that the true difference is zero for {label}."
                )
            else:
                direction = "positive" if bs.observed_diff > 0 else "negative"
                st.success(
                    f"The bootstrap CI **does not include zero**, confirming a "
                    f"{direction} difference in {label} between the two groups."
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Power & Sample Size
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("⚡ Power & Sample Size")

    st.markdown(
        "**Statistical power** is the probability of detecting a real difference when "
        "one exists. A test with 80% power means that if the treatment truly works, "
        "we have an 80% chance of catching it. Low power → lots of false negatives."
    )

    pw_cols = st.columns(2)
    for col, metric in [(pw_cols[0], "retention_1"), (pw_cols[1], "retention_7")]:
        pw: PowerResult = results[f"power_{metric}"]
        label = "1-Day Retention" if metric == "retention_1" else "7-Day Retention"
        achieved = getattr(pw, "achieved_power", None)
        with col:
            st.subheader(label)
            if achieved is not None:
                pct = achieved * 100
                color = "#06D6A0" if pct >= 80 else "#FFD166" if pct >= 60 else "#EF476F"
                st.markdown(
                    f"""<div style="
                        background:{color}22; border-left:5px solid {color};
                        border-radius:6px; padding:12px 16px; margin-bottom:10px;
                    ">
                        <b style="font-size:1.3rem; color:{color};">
                            Achieved Power: {pct:.1f}%
                        </b>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with st.expander("Technical details", expanded=False):
                zr = results[f"ztest_{metric}"]
                st.markdown(
                    f"- Control rate: **{zr.control_rate*100:.2f}%**\n"
                    f"- Treatment rate: **{zr.treatment_rate*100:.2f}%**\n"
                    f"- MDE observed: **{abs(zr.treatment_rate - zr.control_rate)*100:.3f} pp**\n"
                    f"- α = **{alpha:.2f}**, Sample (each group) ≈ "
                    f"**{report['control_size']:,}**"
                )

    st.divider()

    # ── Power curve ───────────────────────────────────────────────────────────
    st.subheader("Power Curve")
    mde_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    pw_metric = st.selectbox(
        "Metric for power curve",
        ["retention_7", "retention_1"],
        key="power_curve_metric",
    )
    zr_pw = results[f"ztest_{pw_metric}"]
    baseline_rate_pw = zr_pw.control_rate
    fig_power = plot_power_curve(
        baseline_rate=baseline_rate_pw,
        mde_values=mde_values,
        sample_size=report["control_size"],
        alpha=alpha,
        metric=pw_metric,
    )
    st.plotly_chart(fig_power, use_container_width=True)

    st.divider()

    # ── Interactive sample size calculator ────────────────────────────────────
    st.subheader("🧮 Sample Size Calculator")
    st.markdown(
        "Use this tool to plan **future experiments**. Enter your assumptions and "
        "see how many users you need in each group."
    )

    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    base_rate = sc_col1.number_input(
        "Baseline Rate (%)",
        min_value=1.0, max_value=99.0,
        value=float(round(zr7.control_rate * 100, 2)),
        step=0.5,
    ) / 100.0
    mde_input = sc_col2.number_input(
        "Minimum Detectable Effect (pp)",
        min_value=0.1, max_value=20.0,
        value=1.0, step=0.1,
    ) / 100.0
    alpha_input = sc_col3.number_input(
        "Alpha (significance)",
        min_value=0.01, max_value=0.20,
        value=alpha, step=0.01,
    )
    power_input = sc_col4.number_input(
        "Desired Power",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.05,
    )

    try:
        req_n = sample_size_calculator(
            baseline_rate=base_rate,
            mde=mde_input,
            alpha=alpha_input,
            power=power_input,
        )
        st.markdown(
            f"""
            <div style="
                background:#4361EE22; border-left:6px solid #4361EE;
                border-radius:8px; padding:16px 20px; margin-top:12px;
            ">
                <h3 style="color:#4361EE; margin:0 0 4px 0;">
                    Required sample size per group: <b>{req_n:,}</b>
                </h3>
                <p style="margin:0;">
                    Total users needed (both groups): <b>{req_n*2:,}</b><br/>
                    Assuming current traffic of ~{report['n_after']:,} users per experiment period,
                    you'd need roughly
                    <b>{max(1, round(req_n*2 / report['n_after'], 1))}</b> experiment period(s).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Could not compute sample size: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Peeking Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("👀 Peeking Simulation")

    st.markdown(
        """
        **The peeking problem** occurs when analysts check experiment results repeatedly
        *while the experiment is running* and stop as soon as they see significance.
        This massively inflates the false-positive rate — you can "find" an effect that
        doesn't exist just by checking at the right moment.

        The chart below simulates experiments where the **true effect is zero** (no real
        difference between groups) and shows how often you'd falsely declare significance
        if you peeked at regular intervals.
        """
    )

    fig_peek = plot_peeking_simulation(results["peeking"])
    st.plotly_chart(fig_peek, use_container_width=True)

    peeking_result = results["peeking"]
    final_fpr = getattr(peeking_result, "final_false_positive_rate", None)
    nominal_fpr = alpha

    if final_fpr is not None:
        st.warning(
            f"**With peeking:** False positive rate ≈ {final_fpr*100:.1f}%  \n"
            f"**Without peeking (nominal):** {nominal_fpr*100:.0f}%  \n"
            f"Peeking inflated the false-positive rate by "
            f"**{final_fpr/nominal_fpr:.1f}×**!"
        )

    st.divider()
    st.subheader("This Experiment Was Not Stopped Early")
    st.success(
        "✅ The Cookie Cats experiment was run to its pre-determined sample size and "
        "was not stopped early. Results are therefore valid under the frequentist framework."
    )

    st.subheader("💡 Practical Tips to Avoid Peeking")
    st.markdown(
        """
        1. **Pre-register** your sample size and stopping rule before the experiment starts.
        2. Use **sequential testing** methods (e.g. SPRT or always-valid p-values) if you
           truly need to monitor results continuously.
        3. Set a **fixed analysis date** — look at results only once, at the end.
        4. If you must peek, apply a **Bonferroni correction** for each peek point.
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Segments
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("🎯 Segment Analysis")

    st.warning(
        "⚠️ **P-hacking risk:** Analysing many subgroups without pre-registering them "
        "increases the risk of finding spurious significant results. Treat segment results "
        "as **exploratory / hypothesis-generating**, not confirmatory."
    )

    seg_metric = st.selectbox(
        "Select metric for segment comparison",
        ["retention_7", "retention_1"],
        key="seg_metric",
    )

    segments_df: pd.DataFrame = results["segments"]
    seg_filtered = segments_df[segments_df["metric"] == seg_metric].copy()

    fig_seg = plot_segment_comparison(seg_filtered, metric=seg_metric)
    st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Segment-Level Rates")

    display_df = seg_filtered[
        ["segment", "gate_30_rate", "gate_40_rate", "gate_30_n", "gate_40_n"]
    ].copy()
    display_df["gate_30_rate"] = display_df["gate_30_rate"].map(lambda x: f"{x*100:.2f}%")
    display_df["gate_40_rate"] = display_df["gate_40_rate"].map(lambda x: f"{x*100:.2f}%")
    display_df.columns = [
        "Segment",
        "gate_30 Rate",
        "gate_40 Rate",
        "gate_30 N",
        "gate_40 N",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Why do zero-round players have 0% retention?")
    st.markdown(
        """
        Players who played **0 rounds** installed the game but never actually played it.
        Since retention is measured as *coming back to play the next day / after 7 days*,
        a player who never started cannot be retained — hence 0% retention is expected
        and correct for this segment. These users likely had technical issues, changed
        their mind, or the game crashed on install.
        """
    )

    # ── Zero-round comparison chart ───────────────────────────────────────────
    with st.expander("Zero-round players breakdown", expanded=False):
        fig_zero = plot_zero_round_comparison(df)
        st.plotly_chart(fig_zero, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Business Impact Calculator
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("💰 Business Impact Calculator")

    st.markdown(
        "Translate the statistical result into **business numbers**. "
        "How many more (or fewer) players do we retain if we ship gate_40?"
    )

    bi_col1, bi_col2 = st.columns(2)
    monthly_installs = bi_col1.number_input(
        "Monthly Installs",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=1_000,
        format="%d",
    )
    baseline_ret = bi_col2.number_input(
        "Baseline 7-Day Retention Rate (%)",
        min_value=0.1,
        max_value=99.9,
        value=float(round(zr7.control_rate * 100, 2)),
        step=0.1,
    )

    # ── Compute impact ────────────────────────────────────────────────────────
    lift_r7_abs = getattr(zr7, "absolute_lift", zr7.treatment_rate - zr7.control_rate)
    treatment_ret = baseline_ret / 100.0 + lift_r7_abs

    players_control = monthly_installs * (baseline_ret / 100.0)
    players_treatment = monthly_installs * treatment_ret
    diff_players = players_treatment - players_control
    rel_change = diff_players / players_control * 100 if players_control else 0.0

    st.divider()

    imp_c1, imp_c2, imp_c3 = st.columns(3)
    imp_c1.metric(
        "Players retained (gate_30 rate)",
        f"{players_control:,.0f}",
    )
    imp_c2.metric(
        "Players retained (gate_40 rate)",
        f"{players_treatment:,.0f}",
        delta=f"{diff_players:+,.0f} ({rel_change:+.1f}%)",
        delta_color="inverse" if diff_players < 0 else "normal",
    )
    imp_c3.metric(
        "Net Δ Retained Players / Month",
        f"{diff_players:+,.0f}",
        delta_color="off",
    )

    # ── Impact bar chart ──────────────────────────────────────────────────────
    fig_impact = go.Figure(
        data=[
            go.Bar(
                x=["gate_30 (Control)", "gate_40 (Treatment)"],
                y=[players_control, players_treatment],
                marker_color=[CONTROL_COLOR, TREATMENT_COLOR],
                text=[f"{players_control:,.0f}", f"{players_treatment:,.0f}"],
                textposition="outside",
            )
        ]
    )
    fig_impact.update_layout(
        title=f"Estimated 7-Day Retained Players per Month<br>(out of {monthly_installs:,} installs)",
        yaxis_title="Retained Players",
        xaxis_title="Group",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    # ── Summary sentence ──────────────────────────────────────────────────────
    if diff_players < 0:
        st.error(
            f"📉 If we ship gate_40, we expect to **lose {abs(diff_players):,.0f} players per month** "
            f"({abs(rel_change):.1f}% fewer retained players) out of {monthly_installs:,} monthly installs."
        )
    elif diff_players > 0:
        st.success(
            f"📈 If we ship gate_40, we expect to **gain {diff_players:,.0f} players per month** "
            f"({rel_change:.1f}% more retained players) out of {monthly_installs:,} monthly installs."
        )
    else:
        st.info("No measurable difference in retained players between the two gates.")

    with st.expander("💡 Assumptions & Caveats", expanded=False):
        st.markdown(
            f"""
            - Retention rate assumed constant across months.
            - Control (gate_30) 7-day retention: **{zr7.control_rate*100:.2f}%**
            - Treatment (gate_40) 7-day retention: **{zr7.treatment_rate*100:.2f}%**
            - Absolute lift: **{lift_r7_abs*100:+.3f} pp**
            - This is a **linear extrapolation** — real-world effects may vary due to
              seasonality, game updates, and user acquisition quality.
            - The lift estimate has uncertainty (see Bootstrap and Bayesian tabs for CIs).
            """
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Built with Streamlit &nbsp;|&nbsp; Cookie Cats A/B Test Analysis "
    "&nbsp;|&nbsp; Data: Mobile Games Industry"
    "</div>",
    unsafe_allow_html=True,
)
