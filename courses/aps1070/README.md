# APS1070H — Foundations of Data Analytics and Machine Learning

Completed Winter 2024. Portable adaptations of four recovered project notebooks.

| Program | What runs | Evaluation |
| --- | --- | --- |
| `aps1070-basic-models` | K-nearest neighbours and decision trees on the body-fat dataset | Training-derived median threshold, five-fold parameter search, held-out accuracy/F1; density removed because it directly determines body-fat estimates |
| `aps1070-satellite-anomalies` | Gaussian mixtures fitted to normal Landsat observations | Separate training, validation and test splits; component count selected by validation AUC; binary `isanomaly` label retained |
| `aps1070-sales-pca` | Standardization, PCA and reconstruction of daily sales by product family | Days are rows, families are columns; ordered 80/20 split; scaling/PCA learned on training observations |
| `aps1070-aileron-regression` | Analytical least squares and batch gradient descent | Training-only scaling and a held-out RMSE comparison |

Run any program with `python run.py <program>`, or use the [course notebook](../../notebooks/aps1070.ipynb). [Source](../../projects/aps1070.py).

## Changes from the original notebooks

Fixed input snapshots replace remote downloads. Evaluation preprocessing is kept inside training splits. Stable least squares replaces explicit matrix inversion. The PCA workflow explicitly transposes the sales input before treating dates as observations. The original GitHub exercise, credentials, Colab uploads and assignment boilerplate are omitted.

These programs preserve the main learning objectives and produce fresh results. They do not replay every original hyperparameter sweep or visualization. The satellite exercise concerns spectral-feature anomaly detection; it is not a street-map, routing, or GIS project.
