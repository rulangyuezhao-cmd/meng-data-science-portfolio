# Reproducibility

## Execution contract

The reference environment is Python 3.12 with the versions in `requirements.txt`. `python run.py all` executes 15 programs and exits nonzero if any fail. Results are written to `generated/`, which is excluded from Git. GitHub Actions runs the input tests, all programs, and result checks, then uploads the generated files as a run artifact.

`data/manifest.json` records snapshot row counts, schemas, SHA-256 checksums and transformations. Randomized algorithms use seed 42. Minor floating-point differences across operating systems and BLAS libraries are expected; the checks test analytical invariants rather than exact model coefficients.

The four course notebooks call the same functions as the command-line runner. This keeps notebook and script behavior aligned. They can run in an installed Python environment without account credentials. The notebooks do not install packages or download data automatically.

## What is verified

- Snapshot integrity, label values, complete skill features, contiguous housing dates, and the orientation of the sales matrix.
- Salary classification excludes raw and encoded versions of its target.
- Every advertised local program completes and produces finite JSON metrics.
- Full-component PCA reconstructs its input; gradient descent approaches the least-squares reference; SQL groups account for observed customer-year values.
- Housing optimization must converge after price scaling; models are assessed against a seasonal baseline.

The checked-in [validation report](validation/local-run.json) records the local execution. The Actions badge reflects execution of the current GitHub version. Passing checks establishes that the advertised workflows run and satisfy these checks; it does not establish unbiased causal estimates, deployment readiness, or investment performance.

## Original work versus portable adaptations

Original notebooks were inspected and used to identify datasets, methods, saved outputs and project scope. The public programs were refactored with AI assistance in November 2026 and corrected where needed. Original files containing credentials, student identifiers, personal reflections, private paths, or third-party assignment text are not published.

The retail demo uses synthetic customer data, and its SQL analysis is a new companion. Skills clustering starts from a recovered aggregate output. Finance programs cover CPU baselines rather than the complete original neural-network model suite. The full retail monthly-feature pipeline, SHAP/LIME runs, Prophet experiments, live scraping and original cloud sessions are not represented as reproduced.

ReasonEval remains a separate repository with its own dependencies and validation limits. This repository's 15-program count excludes ReasonEval. Non-programming courses and the company MEng project have summaries rather than artificial executable substitutes.
