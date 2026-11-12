# APS1070H — Foundations of Data Analytics and Machine Learning

**Course term: Winter 2024 (January–April 2024).**

Portable adaptations of four recovered project notebooks.

## Project questions, results and methods

Results are from the saved [public adaptation validation](../../validation/metrics.json), not a claim that every original experiment was reproduced.

### Classification baselines

**Question**  
Can body measurements classify observations relative to the training-set median body-fat threshold?

**Result**  
On 63 held-out observations, both models reached 69.8% accuracy. KNN F1 was 0.678 and decision-tree F1 was 0.612.

**Methods**

- **Machine learning — classification:** k-nearest neighbours and decision trees.
- **Model selection:** training-only scaling and stratified five-fold grid search.
- **Model evaluation:** held-out accuracy and F1; density excluded to reduce target leakage.

Run: `python run.py aps1070-basic-models`.

### Satellite anomaly detection

**Question**  
Can a model fitted to normal spectral observations distinguish anomalous observations?

**Result**  
A two-component model reached held-out ROC AUC 0.982. The test set contained only 14 anomalies, which limits precision of the estimate.

**Methods**

- **Machine learning — anomaly detection:** Gaussian mixture density modelling.
- **Model selection:** component-count selection on a separate validation set.
- **Model evaluation:** held-out ROC AUC.

Run: `python run.py aps1070-satellite-anomalies`.

### Sales dimensionality reduction

**Question**  
How much sales variation can a smaller representation retain?

**Result**  
Ten components captured 90.2% of training variance across 33 product families; held-out standardized reconstruction MSE was 0.348.

**Methods**

- **Machine learning — dimensionality reduction:** PCA and inverse-transform reconstruction.
- **Data preprocessing:** date/product reshaping and training-only standardization.
- **Model evaluation:** explained variance and held-out reconstruction error.

Run: `python run.py aps1070-sales-pca`.

### Aileron regression

**Question**  
Can iterative optimization match the analytical linear-regression solution?

**Result**  
On 2,750 test rows, RMSE was 0.00016865 for least squares and 0.00016816 for gradient descent.

**Methods**

- **Machine learning — regression:** ordinary least squares.
- **Optimization:** batch gradient descent.
- **Model evaluation:** held-out RMSE after training-only scaling.

Run: `python run.py aps1070-aileron-regression`.

Run any program with `python run.py <program>`, or use the [course notebook](../../notebooks/aps1070.ipynb). [Source](../../projects/aps1070.py).

## Changes from the original notebooks

Fixed input snapshots replace remote downloads. Evaluation preprocessing is kept inside training splits. Stable least squares replaces explicit matrix inversion. The PCA workflow explicitly transposes the sales input before treating dates as observations. The original GitHub exercise, credentials, Colab uploads and assignment boilerplate are omitted.

These programs preserve the main learning objectives and produce fresh results. They do not replay every original hyperparameter sweep or visualization. The satellite exercise concerns spectral-feature anomaly detection; it is not a street-map, routing, or GIS project.
