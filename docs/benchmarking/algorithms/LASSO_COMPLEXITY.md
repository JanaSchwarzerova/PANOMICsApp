# PANOMICs: LASSO computational complexity (implementation-specific)

**Status:** Analytical estimate based on the supplied MATLAB `LASSO(app)` function. **Not an empirical runtime or peak-memory benchmark.**

## Implementation inspected

The application creates a random 70/30 holdout (`cvpartition(n,'HoldOut',0.3)`) and trains on approximately 0.7*n samples. The MATLAB `lasso` call uses `Alpha = app.LASSO_alfa` and `CV = app.LASSO_cv`, selects `FitInfo.Index1SE`, and predicts on the held-out observations. In the first-time **univariate** branch, the same `lasso(XTrain,yTrain,...)` call appears **twice** consecutively, with the first result overwritten; in the first-time multivariate branch it appears once. Reuse branches retrieve stored coefficients instead of fitting. The univariate first-time and multivariate first-time branches compute SHAP-like linear contributions `(XTest - mean(XTrain,1)).*coef'`, store them, and calculate feature importance. The multivariate reuse branch does not compute these contributions.

## Notation and assumptions

- n: number of input samples; p: number of input predictors (SNPs/metabolites).
- n_tr ≈ 0.7n, n_te ≈ 0.3n.
- K: number of cross-validation folds (`app.LASSO_cv`, if a numeric fold count).
- L: number of regularization values along the lambda path (chosen internally by MATLAB unless specified).
- I: effective optimization iterations per lambda and fold; strongly dependent on convergence, tolerance, feature correlations, sparsity, and warm starts.
- Dense double-precision predictors are assumed for illustrative storage calculations; sparse matrices have different costs.

## Time-complexity model

A conservative *work proxy*, not a strict complexity theorem for MATLAB's optimized implementation, is

`W_fit ∝ K × L × I × n_tr × p`.

Cross-validation refits on subsets, so this proxy is intentionally approximate; implementation-dependent final full-training fits and screening can change costs. If K, L and I remain roughly constant, the leading proxy scales as `O(n p)` for fixed hyperparameters. The total first-run cost additionally includes holdout slicing, test prediction and linear contribution calculations, each `O(n p)`.

**PANOMICs-specific duplicate:** the initial univariate path calls `lasso` twice with identical inputs and options. Thus its fitting-work proxy is approximately `2 × W_fit`, whereas the initial multivariate path uses `1 × W_fit`. This is not a benchmarked 2× elapsed-time claim. The duplicate call could be removed without changing the selected results, subject to reproducibility checks because fold assignment may differ between calls when the RNG is not fixed.

**Cached path:** when stored coefficients are reused, the expensive fit is skipped; prediction and contribution calculation scale approximately `O(n p)`. Cached coefficients are not necessarily associated with the newly sampled holdout, so the data split/model provenance should be checked before interpreting test performance.

## Space-complexity model

- Input predictor matrix `X`: `8 n p` bytes for a dense `double` matrix (data payload only).
- Training and test slices together: approximately another `8 n p` bytes if both are materialized as independent dense arrays.
- Full coefficient path `B`: approximately `8 p L` bytes (if dense).
- Held-out SHAP-like contribution matrix: approximately `8 n_te p` bytes.
- Cross-validation, solver workspaces, MATLAB memory overhead, cached app fields and graphics can add significant memory not captured by these estimates.

A useful **lower-bound-style storage accounting** for simultaneously retained dense arrays is `8 × (2 n p + p L + n_te p)` bytes, when X, both split copies, coefficient path and contributions coexist. This is **not peak RAM** and is not guaranteed to equal live allocation at any one instant.

## Illustrative matrix storage (not measured)

| n | p | X alone (MiB) | X + train/test copies + test contributions (MiB) |
|---:|---:|---:|---:|
| 239 | 249 | 0.454 | 1.045 |
| 500 | 1,000 | 3.815 | 8.774 |
| 1,000 | 5,000 | 38.147 | 87.738 |
| 5,000 | 20,000 | 762.939 | 1,754.761 |

The table excludes `B`, solver workspaces and all other allocations. Values assume `n_te ≈ 0.3n`, dense doubles, and 1 MiB = 2^20 bytes.

## Implementation observations relevant to reproducibility

1. Initial univariate training fits the same model twice; the first output is overwritten. This increases work and can change the selected fold realization if cross-validation partitions are randomized.
2. `cvpartition` is generated afresh even in cached-model branches; cached coefficients may have been fitted using a different training subset.
3. The linear contributions labeled SHAP use a training-mean baseline and coefficient multiplication. They are exact additive contributions relative to that baseline for the linear predictor, but should not be described as general conditional SHAP values.
4. The snippet does not show preprocessing. Whether preprocessing was performed within training folds cannot be established from this function alone.
5. The code does not reveal numeric K, L, I, data types, hardware, MATLAB Runtime configuration or web-server quotas. Consequently no elapsed seconds or peak memory can be inferred.

## Suggested GitHub disclosure

> This report describes an implementation-specific analytical work and storage model for the PANOMICs LASSO module. Complexity estimates are conditional on the regularization path, cross-validation folds, solver convergence and dataset representation. The figures and storage calculations are theoretical and should not be interpreted as measured execution times or peak resident memory. Empirical desktop and web benchmarks will be reported separately after instrumented execution.
