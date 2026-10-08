# PANOMICs — Gaussian Process Regression (GPR): Implementation-Aware Computational Complexity

**Status:** Theoretical, code-informed analysis; **not** empirical runtime or peak-memory benchmarking.

## 1. Scope and implementation

The supplied `GPR(app)` routine uses `fitrgp(X,Y)` with MATLAB defaults, followed by `predict` returning a point prediction, prediction standard deviation, and prediction interval. For each of five manually constructed train/test splits, the routine additionally calculates permutation feature importance (`app.calculatePermutationImportance`) and SHAP (`app.calculateSHAP`). The univariate and multivariate branches have substantially similar computational structure; multivariate output is reduced to a row-wise mean before fitting. The code stores the last fitted GPR model and aggregates feature importance across splits. **No empirical timing or memory values have been collected.**

Let **n** be total observations, **p** the number of predictors, **K=5** the number of splits, **m ≈ 0.8n** the training size per split, and **t ≈ 0.2n** the test size per split. Let **q** be the number of full test-set prediction passes used by permutation importance, **s** the number of prediction evaluations per SHAP explanation (which depends on the unseen `calculateSHAP` implementation), and **h** the effective number of model-fit objective/factorization evaluations due to parameter optimization.

## 2. Time complexity

For a *dense exact* Gaussian process with a general covariance kernel, building a training kernel matrix requires approximately **O(m²p)** operations for kernels involving p-dimensional distances, and a dense factorization requires **O(m³)**. Hyperparameter optimization may require repeated evaluations; a useful schematic model is

`T_fit ≈ O(h × (m²p + m³))`.

For t test observations, exact GP predictive mean typically requires kernel evaluations **O(tmp)** and matrix products **O(tm)**; predictive variances/intervals may add **O(tm²)** depending on the factorization and algorithm. A conservative reference envelope for joint mean/uncertainty prediction is

`T_predict ≈ O(tmp + tm²)`.

This is **not a guarantee for `fitrgp`**: MATLAB can select computational options or approximations that change complexity. The source code does not specify `FitMethod`, `PredictMethod`, kernel, active-set options, or hyperparameter search, so their effective behavior must be inspected in the fitted object and MATLAB documentation for the installed release.

### End-to-end PANOMICs workload

The application runs five model fits plus a baseline prediction, permutation importance, and SHAP in each split. An informative accounting identity is:

`T_total = Σ_{k=1..5} [T_fit,k + T_predict,k + T_permutation,k + T_SHAP,k + T_other,k]`.

If permutation importance makes q full-test prediction calls, `T_permutation ≈ q × T_predict` (plus shuffling/scoring overhead). A common one-per-feature approach has q proportional to p, but **the actual q is unknown until `calculatePermutationImportance` is supplied**. SHAP costs cannot be reduced to a credible numeric asymptotic bound without examining `calculateSHAP` (background sample count, evaluation count, batching, and whether an explainer is trained). Thus, the complete application may scale substantially worse than the core GPR model.

## 3. Memory complexity

For dense exact GPR, an m-by-m covariance matrix alone requires **8m² bytes** when represented as a MATLAB double array. Input X requires **8np bytes**. A schematic memory envelope is

`M ≈ O(m² + np + pK + tp + auxiliary SHAP/permutation workspace)`.

The MATLAB process peak can be higher because kernel matrices, factorizations, training/test copies, SHAP background data, and intermediate arrays may coexist. **This envelope is not peak RSS or a server memory quota.**

### Illustrative matrix-storage calculations

These are deterministic storage sizes, not observed application memory:

| n | p | X dense double (MiB) | GP covariance for m = floor(0.8n) (MiB) |
|---:|---:|---:|---:|
| 239 | 249 | 0.454 | 0.278 |
| 500 | 1,000 | 3.815 | 1.221 |
| 1,000 | 5,000 | 38.147 | 4.883 |
| 5,000 | 20,000 | 762.939 | 122.070 |

These figures exclude MATLAB object overhead, temporary copies, factorization workspace, cross-validation buffers, and interpretability calculations.

## 4. Implementation-specific findings (must not be mistaken for benchmark measurements)

1. **Manual split boundaries overlap.** The test indices for iterations i>1 start at `floor(n/5)*(i-1)` rather than `+1`; training indices also include the end-of-test boundary. This can put observations in both training and testing, invalidating leakage-free fold evaluation. Use `cvpartition(n,'KFold',5)` and `training(c,i)` / `test(c,i)` instead.
2. **Fold coverage is uneven.** For n not divisible by five, the final remainder is not assigned to a test fold by the displayed indexing; split sizes are inconsistent.
3. **`MeanSHAPAll` remains zeros.** It is initialized but never assigned `mean(SHAPValuesFold,1)'`, so its reported mean is not informative.
4. **Multivariate result destinations are inconsistent.** Some `PredictionSD` and aggregation assignments target `app.GPR_univar_mat` within multivariate branches; review those destinations.
5. **Multivariate target aggregation.** `fitrgp(X,mean(Y,2))` fits a single model for the row-wise mean, not a multi-output GP.
6. **SHAP and permutation complexity unknown.** The helper function bodies have not been provided; any exact end-to-end complexity or count of model evaluations would be speculative.
7. **Caching/reuse.** The shown branches repeat model fitting even when `app.GPR_univar_mat` or `app.GPR_multivar_mat` is populated; no model-reuse shortcut is apparent in this function.
8. **Only last model retained.** `app.GPR_*_mat.gprMdl1 = gprMdl1` stores the last split's model, not an ensemble of all five.

## 5. Proposed empirical benchmark (future)

Measure **training**, **uncertainty prediction**, **permutation importance**, and **SHAP** separately using `tic/toc` (or `timeit` for standalone functions) and record n, p, fold, MATLAB release, fit/predict method, kernel, helper configuration, CPU, RAM, and OS. Measure actual process **peak resident memory** with an OS-level sampler. Benchmark the desktop and web deployments independently; web execution may impose resource quotas and concurrency limits. Include success/failure outcomes and out-of-memory behavior. Use identical input datasets and algorithm parameters for comparability.

## 6. Appropriate interpretation

This analysis supports a **theoretical discussion of expected computational growth** for the supplied GPR workflow. It does not establish practical maximum dataset sizes, measured seconds, measured peak memory, web capacity, or experimentally validated scalability. Empirical measurements must be added before claiming quantitative software benchmarking.
