# APS1052H — Artificial Intelligence in Finance: From Neural Networks to Deep Learning

Completed Winter 2026. The recovered collection includes coursework in financial time series and a **team project** relating futures positioning to exchange rates. Group-project ideas and datasets are acknowledged as collaborative work; the portable implementations here were prepared for this portfolio.

| Program | Method | Honest comparison |
| --- | --- | --- |
| `aps1052-seasonal-sales` | Log-sales regression with trend, quarterly indicators and time-series tuning | Last eight quarters versus seasonal persistence |
| `aps1052-yield-curve` | Three principal components and a vector autoregression | Last 12 observations versus a last-observation yield-curve baseline |
| `aps1052-cot-forecast` | Ridge and random forest using cached COT/FX features | Ordered holdout and a persistence forecast; external features lagged by one weekly row |
| `aps1052-trading-workflow` | Moving-average signal with a one-day execution lag | Historical equity curve including a 10-basis-point turnover cost |

Run with `python run.py <program>`, or use the [notebook](../../notebooks/aps1052.ipynb). [Source](../../projects/aps1052.py).

## Reproduction boundaries

The public adaptations use fixed historical inputs and run on CPU. They do not reproduce the original TensorFlow neural-network suite, full ARIMAX experiments, all trading strategies, or online market downloads. The trading example is a simplified companion to the recovered strategy coursework.

The course dataset's `CADUSD` column came from Yahoo's `CAD=X` series, representing **CAD per USD**; the legacy column name is retained but its interpretation is corrected. One-row lags reduce timing leakage, but exact COT publication timestamps still need point-in-time validation before any trading use. Hyperparameters use chronological training folds. PCA is fitted on training data and its forecast components are reconstructed without arbitrary sign changes.

The models are compared with simple baselines even when those baselines outperform them. Historical backtest results do not establish trading profitability. The demonstration excludes taxes, financing and market impact.
