# CHE1148H — Process Data Analytics

**Course term: Winter 2025 (January–April 2025).**

Recovered work covers data quality, retail-response models, partial dependence, local explanations, drift, historical country comparisons, and housing time series.

## Project questions, results and methods

Results are from the saved [public adaptation validation](../../validation/metrics.json), not a claim that every original experiment was reproduced.

### Retail model quality

**Question**  
How do data quality and distribution changes affect a customer-response model?

**Result**  
The synthetic public demo reached test ROC AUC 0.654 and F1 0.419 on 400 customers. The deliberately shifted feature distribution had Jensen–Shannon distance 0.534 versus 0.104 for the ordinary holdout. These are demonstration results, not original coursework scores.

**Methods**

- **Machine learning — classification:** class-weighted random forest.
- **Data quality and statistical comparison:** missingness checks, imputation and Jensen–Shannon distance.
- **Model interpretation:** held-out permutation importance and partial dependence.

Run: `python run.py che1148-retail-quality`.

### Retail SQL summaries

**Question**  
How can customer histories be reshaped into consistent annual summaries?

**Result**  
The SQL companion transformed 1,600 synthetic customers into 8,000 customer-year rows and 10 grouped summary rows. It is a new portfolio example, not an original course submission.

**Methods**

- **Data engineering — SQL analytics:** SQLite aggregation and customer-year restructuring.
- **Data validation:** counts and summary consistency checks.

Run: `python run.py che1148-retail-sql`.

### Historical country comparisons

**Question**  
How do historical reported case and death rates compare after population normalization?

**Result**  
The output compares 209 countries using a fixed snapshot through August 27, 2020. Reporting differences limit interpretation; it is not live health or routing analysis.

**Methods**

- **Descriptive statistical analysis:** per-million normalization and continent-level grouping.
- **Data visualization:** comparative plots of historical aggregates.

Run: `python run.py che1148-country-visualization`.

### Toronto housing forecasts

**Question**  
Do seasonal forecasting models beat a simple seasonal-naive baseline?

**Result**  
Across 12 held-out months, seasonal-naive MAE was CAD 22,103, versus CAD 47,772 for Holt–Winters and CAD 46,434 for SARIMA. The simpler baseline won this comparison.

**Methods**

- **Time-series modelling:** Holt–Winters exponential smoothing and SARIMA.
- **Baseline design:** seasonal-naive forecast.
- **Model evaluation:** chronological holdout, MAE and RMSE.

Run: `python run.py che1148-toronto-housing`.

Run with `python run.py <program>`, or open the [notebook](../../notebooks/che1148.ipynb). [Python source](../../projects/che1148.py) · [SQL query](annual_metrics.sql).

## Public-data substitution and method changes

The original retail work used customer transaction features and response labels. The public demo uses 1,600 explicitly fictional customers generated with seed 42. Its scores are **not original coursework performance**, and it does not expose source customer records. The code demonstrates the workflow using annual features; it does not reproduce the complete original monthly feature pipeline, SHAP/LIME explanations, Ridge model, or every sensitivity analysis.

Permutation importance is measured on held-out data. Partial dependence describes model behavior, not causal effects. Drift is deliberately induced and labelled as a simulation. SQL is implemented as an additional portfolio example, not represented as a recovered course submission.

The housing adaptation learns from the first 180 months and evaluates on the last 12. A seasonal naive baseline is included because more complex models need not win. Price scaling improves optimization stability. The original Prophet experiments and live-source downloads are outside this CPU subset.

Country-level comparisons are historical and affected by reporting and population differences. They demonstrate aggregation and visualization rather than public-health conclusions.
