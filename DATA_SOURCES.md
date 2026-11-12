# Data sources and transformations

All inputs are fixed snapshots. Exact files, shapes, transformations and checksums are recorded in [data/manifest.json](data/manifest.json). Dates describe historical course inputs, not current market or health conditions.

| Public file | Source / context | Preparation |
| --- | --- | --- |
| `salary-2024.csv` | [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/2024/), course-cleaned file | Retains salary, work arrangement and education for rows with salary; removes identifiers and all other columns |
| `salary-2022.csv` | [Kaggle 2022 ML and Data Science Survey](https://www.kaggle.com/kaggle-survey-2022), course-cleaned file | Eight categorical predictors and the course-supplied salary bucket; excludes raw salary and its numeric encoding |
| `skill-features.csv` | MIE1624 assignment 3 saved output | Complete 50-skill aggregate table from a displayed notebook cell; no scraped job descriptions |
| `toronto-housing.csv` | [Toronto Dashboard Key Indicators](https://open.toronto.ca/dataset/toronto-s-dashboard-key-indicators/) historical course download | Monthly Average Home Price (City of Toronto), January 2009–December 2024; retains date and value |
| `covid-country-totals.csv` | CHE1148 historical aggregate workbook | Country/date cases, deaths, population and continent through August 2020; underlying provider was not recovered |
| `bodyfat.csv` | [OpenML bodyfat](https://www.openml.org/search?type=data&sort=runs&id=560), version 1 | Fetched from the original notebook's OpenML source; numeric target renamed `target` |
| `landsat.csv` | [APS1070 source dataset](https://github.com/Sabaae/Dataset/blob/master/LandsatSatellite.csv) | Numeric spectral features and binary anomaly label |
| `sales-by-family.csv` | [APS1070 source dataset](https://github.com/Sabaae/Dataset/blob/main/TotalSalesbyFamily.csv) | Families by date; transposed during analysis |
| `ailerons.csv` | [APS1070 source dataset](https://github.com/Sabaae/Dataset/blob/main/Ailerons.csv) | Numeric flight-control regression input and `goal` target |
| `cot-fx.csv` | APS1052 team-project snapshot | Cached weekly COT positions, exchange rate, VIX and Treasury yield inputs; historical `CADUSD` name is interpreted as CAD per USD |
| `treasury-yields.csv` | APS1052 course snapshot | Nine US Treasury maturity series, 2015–2025 |
| `cocacola.csv` | APS1052 course dataset | 61 quarterly sales observations, 1986–2001 |
| `spy.csv` | APS1052 course market snapshot | Historical SPY OHLC, adjusted close and volume |
| `retail-annual.csv` | **Synthetic data created for this portfolio** | 1,600 fictional customers with missing values, annual count/mean/sum features and a simulated response; generator in `tools/generate_retail.py` |

Several legacy course snapshots lack a separately recovered upstream release identifier or license document. Their provenance is labelled accordingly; do not infer a verified commercial-data license from their inclusion here. See [third-party notices](THIRD_PARTY_NOTICES.md). No license over underlying third-party data is granted by this repository.
