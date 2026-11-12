# MIE1624H — Introduction to Data Science and Analytics

**Course term: Fall 2024 (September–December 2024).**

The dates above identify the course session, not the exact working period of one assignment. Three submitted notebooks were recovered: statistical analysis, salary classification, and job-skills clustering.

## Project questions, results and methods

Results are from the saved [public adaptation validation](../../validation/metrics.json), not a claim that every original experiment was reproduced.

### Salary distributions

**Question**  
How do salary distributions differ across work arrangements and education groups?

**Result**  
The public adaptation retained 11,748 of 12,209 observations. The remote-minus-in-person mean difference was about USD 31,572, with a bootstrap 95% interval of USD 29,067–34,117. This is an observational association, not a causal or online experiment result.

**Methods**

- **Statistical analysis — hypothesis testing:** Welch's independent-samples t-test and unequal-variance one-way ANOVA.
- **Statistical analysis — uncertainty estimation:** 10,000 bootstrap resamples and percentile confidence intervals.
- **Data preparation:** missing-value handling, IQR screening and grouped summaries.

Run: `python run.py mie1624-salary-statistics`.

### Salary classification

**Question**  
Can survey features distinguish 15 salary categories?

**Result**  
On 2,034 held-out observations, macro F1 was 0.105 versus 0.036 for the majority-class baseline. The improvement is modest; this is not presented as a production salary estimator.

**Methods**

- **Machine learning — classification:** class-weighted logistic regression and a majority-class baseline.
- **Data preprocessing:** categorical imputation and one-hot encoding.
- **Model evaluation:** stratified cross-validation, macro F1, balanced accuracy and confusion matrices.

Run: `python run.py mie1624-salary-classification`.

### Job-skills clustering

**Question**  
What groups emerge from the recovered feature table for 50 job skills?

**Result**  
The adaptation produced eight groups; the K-means silhouette score was 0.521. Results describe a hand-designed aggregate feature table, not a new scrape of job postings.

**Methods**

- **Machine learning — clustering:** K-means and Ward-linkage agglomerative clustering.
- **Machine learning — dimensionality reduction:** PCA for visualization.
- **Model evaluation:** silhouette score and inspection of cluster assignments.

Run: `python run.py mie1624-skill-clustering`.

[Run as a notebook](../../notebooks/mie1624.ipynb) · [Python source](../../projects/mie1624.py)

## Method and interpretation

The statistical adaptation retains the original 12,209 to 11,748 row IQR filtering result. It resamples observed groups and uses quantiles of the mean-difference distribution for uncertainty. It does **not** treat 10,000 bootstrap replicates as 10,000 independent survey participants. Differences reflect observational associations and may be confounded by geography, role, seniority, or selection into remote work. This was not an online randomized A/B test. IQR filtering also changes the target population, especially high earners.

The classifier uses eight compact predictors: education, coding experience, ML experience, job role, industry, company size, data-team size, and ML adoption. It excludes the raw salary and encoded salary fields. The course's supplied `Q29_buckets` labels are retained. Macro F1 highlights performance across an imbalanced 15-class target; modest scores are reported without presenting this baseline as a production salary estimator.

The clustering input is an aggregate table recovered from a fully displayed notebook output, not reconstructed job descriptions. It contains frequency and nine manually defined skill attributes. Those hand-designed categories influence the resulting clusters. The original scraping, description cleaning, and NER setup are not rerun; original job-posting text is not redistributed.
