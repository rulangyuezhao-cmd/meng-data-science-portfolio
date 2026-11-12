# CHE1148H — Process Data Analytics

Completed Winter 2025. Recovered work covers data quality, retail-response models, partial dependence, local explanations, drift, historical country comparisons, and housing time series.

| Program | What runs | Scope |
| --- | --- | --- |
| `che1148-retail-quality` | Missingness checks, imputation, random forest, permutation importance, partial dependence and Jensen–Shannon drift comparison | **Synthetic public demo**, with a deliberately simulated distribution shift |
| `che1148-retail-sql` | Customer-year reshaping and SQLite aggregation | **New portfolio companion**, demonstrating SQL over the same synthetic input |
| `che1148-country-visualization` | Historical reported cases/deaths per million, grouped by continent | Fixed 2020 country aggregates; no live health information or GIS claims |
| `che1148-toronto-housing` | Seasonal naive, Holt–Winters and SARIMA forecasts | Monthly Toronto home-price snapshot, final 12 months held out |

Run with `python run.py <program>`, or open the [notebook](../../notebooks/che1148.ipynb). [Python source](../../projects/che1148.py) · [SQL query](annual_metrics.sql).

## Public-data substitution and method changes

The original retail work used customer transaction features and response labels. The public demo uses 1,600 explicitly fictional customers generated with seed 42. Its scores are **not original coursework performance**, and it does not expose source customer records. The code demonstrates the workflow using annual features; it does not reproduce the complete original monthly feature pipeline, SHAP/LIME explanations, Ridge model, or every sensitivity analysis.

Permutation importance is measured on held-out data. Partial dependence describes model behavior, not causal effects. Drift is deliberately induced and labelled as a simulation. SQL is implemented as an additional portfolio example, not represented as a recovered course submission.

The housing adaptation learns from the first 180 months and evaluates on the last 12. A seasonal naive baseline is included because more complex models need not win. Price scaling improves optimization stability. The original Prophet experiments and live-source downloads are outside this CPU subset.

Country-level comparisons are historical and affected by reporting and population differences. They demonstrate aggregation and visualization rather than public-health conclusions.
