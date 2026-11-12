# APS1052H — Artificial Intelligence in Finance: From Neural Networks to Deep Learning

**Course term: Winter 2026 (January–April 2026).**

The recovered collection includes coursework in financial time series and a **team project** relating futures positioning to exchange rates. Group-project ideas and datasets are acknowledged as collaborative work; the portable implementations here were prepared for this portfolio.

## Project questions, results and methods

Results are from the saved [public adaptation validation](../../validation/metrics.json), not a claim that every original experiment was reproduced.

### Seasonal sales forecasting

**Question**  
Does a trend-and-season model improve on seasonal persistence?

**Result**  
On eight held-out quarters, model MAE was 1,034.60 in the dataset's sales units, versus 352.50 for the baseline. The baseline performed better.

**Methods**

- **Statistical modelling — regression:** log-sales regression with trend and quarterly indicators.
- **Model selection:** time-series tuning.
- **Model evaluation:** chronological holdout and MAE.

Run: `python run.py aps1052-seasonal-sales`.

### Yield-curve forecasting

**Question**  
Can a compact representation forecast movements across Treasury maturities?

**Result**  
Three components captured 99.8% of training variance. Held-out RMSE was 0.002008 in the stored yield units, versus 0.002284 for persistence.

**Methods**

- **Machine learning — dimensionality reduction:** PCA fitted on training data.
- **Time-series modelling:** vector autoregression (VAR).
- **Model evaluation:** held-out forecast RMSE against persistence.

Run: `python run.py aps1052-yield-curve`.

### COT and exchange-rate forecasting

**Question**  
Do futures-positioning features improve exchange-rate forecasts over persistence?

**Result**  
On 45 test weeks, log-FX RMSE was 0.01146 for Ridge, 0.01585 for random forest and 0.00662 for persistence. Persistence won; feature release timing still needs point-in-time validation.

**Methods**

- **Machine learning — regression:** Ridge and random forest.
- **Feature engineering:** lagged COT/FX features.
- **Model evaluation:** chronological holdout, MAE/RMSE and persistence comparison.

Run: `python run.py aps1052-cot-forecast`.

### Trading workflow demonstration

**Question**  
How does an explicit execution lag and turnover cost affect a simple trading rule?

**Result**  
The fixed historical backtest produced 105.6% total strategy return versus 143.2% for buy-and-hold over the stored period, with a 10-basis-point turnover cost. This educational result is not evidence of future profitability.

**Methods**

- **Time-series analytics:** moving-average signals.
- **Backtesting:** one-day execution lag and turnover-cost accounting.
- **Evaluation:** historical equity curve and buy-and-hold comparison; taxes, financing and market impact excluded.

Run: `python run.py aps1052-trading-workflow`.

Run with `python run.py <program>`, or use the [notebook](../../notebooks/aps1052.ipynb). [Source](../../projects/aps1052.py).

## Reproduction boundaries

The public adaptations use fixed historical inputs and run on CPU. They do not reproduce the original TensorFlow neural-network suite, full ARIMAX experiments, all trading strategies, or online market downloads. The trading example is a simplified companion to the recovered strategy coursework.

The course dataset's `CADUSD` column came from Yahoo's `CAD=X` series, representing **CAD per USD**; the legacy column name is retained but its interpretation is corrected. One-row lags reduce timing leakage, but exact COT publication timestamps still need point-in-time validation before any trading use. Hyperparameters use chronological training folds. PCA is fitted on training data and its forecast components are reconstructed without arbitrary sign changes.

The models are compared with simple baselines even when those baselines outperform them. Historical backtest results do not establish trading profitability. The demonstration excludes taxes, financing and market impact.
