# PANOMICs — Elastic Net Regression (ENR): Implementation-Specific Complexity Analysis

**Status:** Source-verified theoretical analysis of the supplied MATLAB `function ENR(app)`. **Not an empirical runtime or peak-RAM benchmark.**

## 1. Source-verified execution path

- Input: `X=app.x_input` (n observations by p predictors), `y=app.y_output`.
- On first fit, the univariate and multivariate-labelled branches each create `cvpartition(n,'HoldOut',0.3)` (approximately 70% training, 30% testing).
- Each fresh branch calls `[B,FitInfo]=lasso(XTrain,yTrain,'Alpha',app.ENR_alfa,'CV',app.ENR_cv)` **once**. `FitInfo.Index1SE` selects the coefficient vector and intercept. The comment says five-fold CV, but the actual fold count is controlled by `app.ENR_cv`; it is not fixed to five in the function.
- Predictions: `XTest*coef + coef0`.
- Interpretability: `SHAPValues=(XTest-mean(XTrain,1)).*coef'`; stores absolute and signed mean contributions. These are *exact additive linear contributions relative to the training-mean baseline*, not evidence of a model-agnostic SHAP explainer being run.
- Cached branches load `B` and `FitInfo`, create a **new** holdout split, and do not refit. Training time for a cache hit is therefore not the same as a fresh fit.
- No explicit predictor standardization, random seed, or held-out preprocessing is shown here. MATLAB `lasso` defaults and upstream processing must be documented separately.

## 2. Elastic-net objective and identity check

The MATLAB `lasso` `Alpha` parameter controls the mixture of L1 and L2 penalties. The intended elastic-net objective is conventionally represented as

`(1/(2*n_tr))*||y-X*beta||_2^2 + lambda*[alpha*||beta||_1 + (1-alpha)/2*||beta||_2^2]`.

Confirm `0 < app.ENR_alfa < 1` at runtime to substantiate the **elastic-net** label. `Alpha=1` is LASSO, and the actual code does not show the chosen value.

## 3. Analytical time model

Let `n_tr≈0.7n`, `n_te≈0.3n`, `p` predictors, `K=app.ENR_cv` folds (if numeric), `L` fitted lambda values, and `I` effective iterations per lambda/fold. A dense coordinate-descent-style work proxy is

`W_fit ≈ O((K+1)*L*I*n_tr*p)`.

The `+1` represents the refit/full-training path in a simplified cost accounting; MATLAB may use warm starts, screening, optimized solvers, or different fold-specific iteration counts. **This is not an exact operation count for MATLAB `lasso`.** Unlike the supplied LASSO univariate routine, the ENR fresh-fit path does **not** duplicate the `lasso` call.

Additional work: holdout indexing/copies `O(np)`; prediction `O(n_te*p)`; baseline mean `O(n_tr*p)`; linear contributions and aggregation `O(n_te*p)`. Thus:

`W_fresh ≈ W_fit + O(np+n_te*p)`.

Cached execution excludes `W_fit` but still performs new split, prediction, and attribution. The formula cannot be translated to seconds without timing MATLAB on specified hardware.

## 4. Analytical memory model

For dense MATLAB `double` matrices (8 bytes/value):

- Original X: `8*n*p` bytes.
- Holdout training/test copies: together approximately another `8*n*p` bytes, if materialized.
- Full coefficient path B: `8*p*L` bytes, plus FitInfo metadata and solver workspace.
- Test-set contribution matrix SHAPValues: `8*n_te*p` bytes.
- Prediction/response vectors and feature summaries: `O(n+p)`.

A transparent **lower-level storage proxy**, not a measured peak, is

`M_proxy = 8*(2*n*p + n_te*p + p*L) bytes`.

It excludes CV fold buffers, numerical solver state, copies during expression evaluation, MATLAB runtime, UI, and any other matrices held by the application. Asymptotic storage for the shown data and output structures is `O(np+pL)`; peak runtime memory may be higher.

| Scenario | n | p | Original X (GiB) | X + split copies + test contributions (GiB; excludes B and workspace) |
|---|---:|---:|---:|---:|
| Demo | 239 | 249 | 0.00044 | 0.00102 |
| Small | 500 | 1,000 | 0.00373 | 0.00857 |
| Medium | 1,000 | 5,000 | 0.03725 | 0.08568 |
| Large | 5,000 | 20,000 | 0.74506 | 1.71363 |
| High-dimensional | 10,000 | 50,000 | 3.72529 | 8.56817 |

These are decimal-to-binary conversions of double storage arithmetic, **not actual peak RAM** or validated feasibility thresholds.

## 5. Code-level limitations affecting correctness and benchmarking

1. **Cached-model data leakage risk:** a fresh random 70:30 split is generated even when loading previously trained B/FitInfo. Previously used training rows may appear in the new test set. Save/reuse the original partition or evaluate an explicitly independent external dataset.
2. **Multivariate-labelled path is not multi-response:** `y(idxTrain)` and `y(idxTest)` use linear indexing; `lasso` is called with a vector response. For an n-by-q response matrix this does not implement q-output ENR and may select unintended elements. `n=length(y)` can also be wrong for q>n. Specify target aggregation or fit separate response models with correct row indexing.
3. **Randomness:** no fixed RNG seed is shown; repeated runs need reproducible partitions and recorded seeds.
4. **Penalty identity:** the ENR label requires the actual alpha to be strictly between zero and one; record `app.ENR_alfa` and the lambda grid.
5. **Interpretability terminology:** linear baseline contributions are not a general SHAP algorithm; describe the baseline and additive decomposition precisely.
6. **CV and preprocessing:** preprocessing must be fitted on training data only; the source does not establish how earlier transformations are performed.
7. **No empirical results:** CPU runtime, peak RAM, web-server RAM limits, and model throughput have not been measured.

## 6. Suggested empirical benchmarking

Benchmark **fresh training**, **cached prediction**, and **attribution** separately on the same fixed splits and several (n,p) configurations; record actual `app.ENR_alfa`, `app.ENR_cv`, lambda count, MATLAB version, hardware, wall time, peak process RAM, and any failures. Run desktop and web deployment separately. Repeat 3–5 times and report variability.

## 7. Manuscript-ready qualification

> Elastic Net Regression in PANOMICs is implemented using MATLAB's regularization-path `lasso` routine with an application-defined mixing parameter and cross-validation setting. A 70:30 holdout split is used for model evaluation, while the one-standard-error rule selects the regularization level. For dense inputs, a coordinate-descent-style analytical workload proxy scales with the number of training samples, predictors, regularization values, optimization iterations, and internal validation folds. Stored input and coefficient-path arrays scale approximately as O(np+pL). These relations are theoretical and do not represent measured desktop or web execution times or peak memory utilization.
