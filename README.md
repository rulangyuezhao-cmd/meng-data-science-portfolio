# MEng Data Science Portfolio

**Chelsea Rulangyue Zhao · University of Toronto**  
Master of Engineering, Chemical Engineering · Emphasis in Data Analytics and Machine Learning · Degree conferred October 2026

[![Run portfolio](https://github.com/rulangyuezhao-cmd/meng-data-science-portfolio/actions/workflows/validate.yml/badge.svg)](https://github.com/rulangyuezhao-cmd/meng-data-science-portfolio/actions/workflows/validate.yml)
[LinkedIn](https://www.linkedin.com/in/zhao-chelsea/) · [GitHub profile](https://github.com/rulangyuezhao-cmd) · [ReasonEval](https://github.com/rulangyuezhao-cmd/mie1512-reasoneval-portfolio)

This collection connects my graduate coursework to practical data science: preparing data, testing hypotheses, comparing models, interpreting results, and building repeatable analytical workflows.

**15 runnable CPU analyses across four technical courses**, plus the separate ReasonEval repository and summaries of my remaining completed MEng courses. The programs here are compact portfolio adaptations of recovered coursework, prepared with AI-assisted refactoring in November 2026. They document changes from the original submissions; they are not exact replays of every original notebook or model experiment.

Course entries use the format **course code + recorded title + academic term + month range**. For example, MIE1624H was **Fall 2024 (September–December)** and MIE1512H was **Winter 2026 (January–April)**. See the [verified course timeline](COURSE_COVERAGE.md); session dates are distinct from assignment dates and later portfolio reconstruction.

## Choose a project

Each project page separates **Question**, **Result**, and **Methods**. Methods are grouped by purpose—such as statistical hypothesis testing, machine-learning classification, clustering, or time-series modelling—rather than mixed into the result.

| Area | Projects | Explore |
| --- | --- | --- |
| Statistical analysis and machine learning | Salary comparisons; salary classification; job-skills clustering | [MIE1624H · Fall 2024](courses/mie1624/) |
| Machine-learning foundations | Classification baselines; spectral anomaly detection; sales PCA; aileron regression | [APS1070H · Winter 2024](courses/aps1070/) |
| Process analytics and forecasting | Retail model quality; SQL summaries; historical country comparisons; housing forecasts | [CHE1148H · Winter 2025](courses/che1148/) |
| Financial modelling | Seasonal sales; yield curves; COT/FX forecasts; trading workflow | [APS1052H · Winter 2026](courses/aps1052/) |
| LLM evaluation | ReasonEval inference and reasoning-error analysis | [MIE1512H · Winter 2026](courses/mie1512/) |

### Example: salary statistics

**Question**  
How do observed salaries differ by work arrangement and education?

**Result**  
The public adaptation retained 11,748 of 12,209 observations. Its remote-minus-in-person mean difference was about USD 31,572, with a bootstrap 95% interval of USD 29,067–34,117. These observational comparisons do not establish causal effects.

**Methods**

- **Statistical analysis — hypothesis testing:** Welch's t-test and unequal-variance ANOVA.
- **Statistical analysis — uncertainty estimation:** bootstrap resampling and confidence intervals.
- **Data preparation:** IQR screening and grouped summaries.

[Open the full project and limitations →](courses/mie1624/)

## Run the projects

Python **3.12** is the reference environment. No cloud account, API key, GPU, or private Drive mount is required for these 15 programs. Package installation requires internet access; the analyses use bundled inputs.

```bash
git clone https://github.com/rulangyuezhao-cmd/meng-data-science-portfolio.git
cd meng-data-science-portfolio
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py list
python run.py all
python tools/check_results.py
```

For a single example: `python run.py mie1624-salary-statistics`. Results, CSV tables, and figures appear in `generated/<project>/`. To explore interactively, open a [course notebook](notebooks/) in VS Code or Jupyter and select this Python environment. The command-line runner is the simplest way to reproduce everything.

## What the work demonstrates

| Capability | Evidence |
| --- | --- |
| Applied statistics | Welch tests, bootstrap confidence intervals, distribution checks, explicit limits on causal interpretation |
| Data wrangling | Typed CSV inputs, missing-value handling, restructuring time series, aggregate feature tables |
| Machine learning | Pipelines, validation splits, cross-validation, classifiers, regression, anomaly detection, clustering, PCA |
| SQL and processing workflows | SQLite customer-year aggregation, data manifests, repeatable commands, machine-readable result checks |
| Git and reproducibility | Versioned code and inputs, pinned analytical dependencies, GitHub Actions execution |
| Communication | Course context, scope notes, honest model comparisons and documented limitations |

My Azure experience comes from the CMTE industry project with Wolseley Express. These local programs do not demonstrate a deployed Azure service. The collection does not claim Property & Casualty insurance experience, randomized online A/B testing, or production geospatial/routing work.

## Example outputs

**Statistical uncertainty:** a bootstrap distribution of the remote-minus-in-person mean salary difference in the filtered survey sample. This is an observational comparison, not a causal estimate.

![Bootstrap mean-difference distribution](figures/bootstrap-mean-difference.png)

**Forecast evaluation:** model predictions are compared with held-out Toronto home prices and a seasonal naive baseline. The baseline is retained even when it wins.

![Toronto housing forecast comparison](figures/forecast.png)

Current numerical results are in [validation/metrics.json](validation/metrics.json); rerunning creates fresh outputs under `generated/`.

## Education and course coverage

I hold a **B.A.Sc. in Chemical Engineering** (University of Toronto, conferred June 2023), with minors in **Bioengineering** and **Engineering Business**, and a **Master of Engineering in Chemical Engineering** (conferred October 2026).

The [complete MEng course index](COURSE_COVERAGE.md) follows my academic history rather than the existence of a Drive folder. It includes all seven completed taught courses, the MEng project, and the safety workshop. Non-programming courses are represented by summaries. Dropped courses are excluded.

See [reproducibility and validation](REPRODUCIBILITY.md), [data provenance](DATA_SOURCES.md), and [third-party notices](THIRD_PARTY_NOTICES.md) for the exact boundaries of this portfolio.
