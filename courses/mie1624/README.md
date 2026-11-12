# MIE1624H — Introduction to Data Science and Analytics

Completed Fall 2024. Three submitted notebooks were recovered: statistical analysis, salary classification, and job-skills clustering.

| Run command | Input | Output and method |
| --- | --- | --- |
| `python run.py mie1624-salary-statistics` | 12,209 salary observations from the course-cleaned 2024 Stack Overflow survey | IQR filtering; remote/in-person Welch test; education-group Welch ANOVA; 10,000 bootstrap mean differences and a 95% interval |
| `python run.py mie1624-salary-classification` | 8,136 rows from the course-cleaned Kaggle 2022 survey | Categorical imputation/encoding, balanced logistic regression, training-only cross-validation, held-out macro F1 and a majority-class baseline |
| `python run.py mie1624-skill-clustering` | Complete saved 50-skill feature table from assignment 3 | Frequency scaling, K-means and hierarchical clusters, silhouette score and a PCA visualization |

[Run as a notebook](../../notebooks/mie1624.ipynb) · [Python source](../../projects/mie1624.py)

## Method and interpretation

The statistical adaptation retains the original 12,209 to 11,748 row IQR filtering result. It resamples observed groups and uses quantiles of the mean-difference distribution for uncertainty. It does **not** treat 10,000 bootstrap replicates as 10,000 independent survey participants. Differences reflect observational associations and may be confounded by geography, role, seniority, or selection into remote work. This was not an online randomized A/B test. IQR filtering also changes the target population, especially high earners.

The classifier uses eight compact predictors: education, coding experience, ML experience, job role, industry, company size, data-team size, and ML adoption. It excludes the raw salary and encoded salary fields. The course's supplied `Q29_buckets` labels are retained. Macro F1 highlights performance across an imbalanced 15-class target; modest scores are reported without presenting this baseline as a production salary estimator.

The clustering input is an aggregate table recovered from a fully displayed notebook output, not reconstructed job descriptions. It contains frequency and nine manually defined skill attributes. Those hand-designed categories influence the resulting clusters. The original scraping, description cleaning, and NER setup are not rerun; original job-posting text is not redistributed.
