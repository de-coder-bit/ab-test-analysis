# 🍪 Cookie Cats A/B Test Analysis
## A Production-Grade End-to-End A/B Testing Framework

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.13-8CAAE6?logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-4051B5)

---

## 🎯 Business Problem

[Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats) is a hugely popular mobile puzzle game developed by Tactile Entertainment. As players progress through the game, they occasionally encounter **gates** — forced wait timers or in-app purchase prompts designed to give players a break (and drive monetisation).

This experiment tested whether **moving the first gate from level 30 to level 40** improves player retention. The hypothesis: letting players progress further before hitting a gate creates a better first experience. The data tells a different story.

- **N = 90,189 players** randomised into two groups
- **Control (`gate_30`):** Gate remains at level 30 — 44,700 players
- **Treatment (`gate_40`):** Gate moved to level 40 — 45,489 players
- **Primary metric:** 7-day retention
- **Secondary metrics:** 1-day retention, game rounds played

---

## 📊 Key Findings

| Metric | Control (`gate_30`) | Treatment (`gate_40`) | Lift | p-value | Significant? |
|---|---|---|---|---|---|
| 1-Day Retention | 44.82% | 44.23% | -0.59pp (-1.3%) | 0.073 | ❌ No (95%) |
| 7-Day Retention | 19.02% | 18.20% | -0.82pp (-4.3%) | 0.0016 | ✅ Yes |
| Median Game Rounds | 17 | 16 | -1 round (-5.9%) | 0.051 | ❌ No |

### 🏁 Verdict: 🚫 DON'T SHIP `gate_40`

> Moving the gate to level 40 **significantly reduces 7-day retention by 4.3% in relative terms** (p = 0.0016). The Bayesian analysis further confirms this: **P(gate_40 better) ≈ 3.4%** for 7-day retention — meaning there is a 96.6% probability that the control is superior. Keep the gate at level 30.

---

## 🔬 Methodology

A rigorous, multi-layered statistical approach was applied to ensure the conclusions are robust:

- **Frequentist — Two-proportion z-test:** Primary significance test on 1-day and 7-day retention rates, with Holm-Bonferroni multiple comparison correction applied across all metrics. Holm's step-down procedure is uniformly more powerful than vanilla Bonferroni while still controlling the family-wise error rate.
- **Bayesian — Beta-Binomial model:** Modelled each group's retention rate as a Beta distribution and simulated 100,000 posterior samples to compute P(treatment > control). For 7-day retention, P(gate_40 better) = **3.4%**, providing an intuitive probability statement beyond binary significance.
- **Bootstrap — 10,000-resample percentile CI:** Non-parametric confidence intervals constructed via bootstrapping are fully consistent with the frequentist z-test results, strengthening confidence in the findings.
- **Power analysis:** Achieved statistical power of **~82%** for 7-day retention at α = 0.05, meaning the study was adequately powered to detect the observed effect size.
- **Data quality — SRM check:** A Sample Ratio Mismatch (SRM) check was performed to verify randomisation integrity. The observed group split (44,700 vs. 45,489) was tested against the expected 50/50 allocation; no significant SRM was detected (p > 0.05).
- **Outlier handling:** Game rounds were capped at the 99.9th percentile to prevent extreme values from distorting distributional analyses, while retention boolean flags required no capping.
- **Peeking simulation:** A Monte Carlo simulation demonstrated how repeatedly checking p-values during data collection inflates false positive rates — illustrating why pre-registration and fixed sample sizes matter in practice.
- **Segmentation analysis:** Retention was broken down by **engagement tier** (casual, regular, hardcore players by game rounds quartile), revealing that the gate change had a disproportionate negative impact on mid-engagement players.

---

## 📸 Screenshots

| Executive Summary | Bayesian Results | Bootstrap Distribution |
|---|---|---|
| ![exec](screenshots/exec_summary.png) | ![bayes](screenshots/bayesian.png) | ![boot](screenshots/bootstrap.png) |

<!-- TODO: Add screenshots after deployment -->

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

<!-- TODO: Add live link after deployment -->

The interactive Streamlit dashboard features **9 tabs**:
1. **Executive Summary** — headline KPIs and verdict
2. **EDA** — distribution plots and data quality report
3. **Retention Analysis** — frequentist test results with confidence intervals
4. **Bayesian Analysis** — posterior distributions and probability statements
5. **Bootstrap** — resampling CIs and sampling distributions
6. **Power Analysis** — power curves across effect sizes and sample sizes
7. **Multiple Comparisons** — Holm vs. Bonferroni comparison table
8. **Peeking Simulation** — false positive rate under sequential testing
9. **Segmentation** — retention breakdown by engagement tier

---

## 🖥️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cookie-cats-ab-test.git
cd cookie-cats-ab-test

# 2. Create virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
# Put cookie_cats.csv in the data/ directory
# Dataset source: https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats

# 5. Run the dashboard
streamlit run app/streamlit_app.py

# 6. Run tests
pytest tests/ -v

# 7. Open the notebook
jupyter notebook notebooks/01_eda_and_analysis.ipynb
```

---

## 📂 Project Structure

```
ab-test-analysis/
├── data/
│   └── cookie_cats.csv              # Dataset (place here before running)
├── notebooks/
│   └── 01_eda_and_analysis.ipynb    # Full exploratory & statistical analysis
├── src/
│   ├── data_loader.py               # Load, validate, clean & SRM check
│   ├── stats_tests.py               # All statistical functions (frequentist, Bayesian, bootstrap)
│   └── visuals.py                   # Reusable Plotly chart components
├── app/
│   └── streamlit_app.py             # 9-tab interactive dashboard
├── tests/
│   └── test_stats.py                # 35+ pytest test cases
├── reports/
│   └── final_report.md              # Executive summary for stakeholders
├── .streamlit/
│   └── config.toml                  # Dark theme configuration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Core language |
| pandas | 2.2 | Data manipulation and aggregation |
| scipy | 1.13 | Statistical tests (z-test, bootstrap) |
| statsmodels | 0.14 | Power analysis, Holm-Bonferroni correction |
| plotly | 5.24 | Interactive visualisations |
| streamlit | 1.40 | Multi-tab dashboard framework |
| pytest | 8.3 | Unit testing (35+ test cases) |
| numpy | 1.26 | Numerical computing & random sampling |
| jupyter | — | Exploratory analysis notebook |

---

## 🧠 What I Learned

- **End-to-end A/B test workflow:** Structured the full pipeline — data ingestion → quality checks → EDA → hypothesis testing → power analysis → decision — mirroring a real product analytics workflow.
- **Why SRM checks matter:** Randomisation failures silently corrupt results; a chi-squared SRM check is a critical first gate before any statistical analysis.
- **Statistical vs. practical significance:** The 1-day retention difference (p = 0.073) is not significant, yet the 7-day result (p = 0.0016) is — demonstrating that short-term and long-term player behaviour can diverge meaningfully, and that multiple metrics require correction.
- **Multiple comparison correction — Holm > Bonferroni:** Holm's step-down procedure controls the family-wise error rate while rejecting more null hypotheses than Bonferroni; it should be the default choice for multi-metric experiments.
- **Bayesian Beta-Binomial as a complement:** The frequentist p-value answers "is this compatible with the null?" while the Bayesian posterior answers "how likely is the treatment to be better?" — both perspectives together tell a richer story.
- **The peeking problem:** Simulating sequential testing showed that checking p-values daily during an experiment can inflate the false positive rate from 5% to over 20%, reinforcing the importance of pre-specified sample sizes and stopping rules.
- **Production dashboard development:** Built a fully interactive, 9-tab Streamlit app with Plotly charts, real-time parameter widgets, and a dark theme — deployable with a single `streamlit run` command.

---

## 🔒 Data Privacy

The dataset is fully anonymised. Only pseudonymous user IDs (not names or personal information), aggregate gameplay metrics (rounds played), and boolean retention flags (did the player return after 1/7 days?) are used. No personally identifiable information is present.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ and rigorous statistics · <a href="https://github.com/YOUR_USERNAME/cookie-cats-ab-test">GitHub</a>
</p>
