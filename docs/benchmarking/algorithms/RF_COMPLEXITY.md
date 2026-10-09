# PANOMICs — Random Forest (RF): Theoretical Computational Complexity

**Status:** Analytical, implementation-informed assessment; not empirical benchmarking.

## Implementation reviewed

The supplied `RF(app)` MATLAB function trains a regression ensemble with `TreeBagger(T, X, y, Method='regression', OOBPrediction='on')`, where `T = app.number_of_decisionsTrees_input`. For multiple outcomes it trains on `mean(y,2)` rather than independently predicting each outcome. It computes permutation importance using `app.calculateRFPermutationImportance`, SHAP using `app.calculateSHAP(@(Xnew) predict(RF_GM_1,Xnew),X,X)`, and OOB quantile predictions via `oobQuantilePredict`.

No independent held-out split or fivefold cross-validation appears in this RF function. OOB predictions are an internal resampling-based assessment, not an independent external test set.

## Notation

- `n`: number of rows in X; `p`: number of predictors.
- `T`: number of regression trees.
- `m`: number of candidate features evaluated at a split (implementation-dependent).
- `d`: mean tree depth; `s`: mean number of stored nodes per tree.
- `R`: number of permutation repeats per feature (unknown until helper code is reviewed).
- `Q`: total number of model evaluations performed by SHAP (unknown until helper code is reviewed).

## Training time

An illustrative balanced-tree cost model, assuming candidate split evaluation by sorting within nodes, is

`O(T * m * n * log(n)^2)`.

With reusable sorted orders or efficient split-search implementations, a more favorable illustrative bound is `O(T * m * n * log(n))`. These are **conditional algorithmic models**, not measured or guaranteed MATLAB `TreeBagger` complexities. Actual behavior depends on tree growth, minimum leaf size, predictor sampling, and implementation details.

## Prediction and interpretation

A prediction for `n_eval` observations traversing `T` trees of typical depth `d` requires approximately `O(n_eval * T * d)` operations, excluding overhead. OOB quantile predictions have additional aggregation/sorting costs that depend on the internal implementation.

For a helper that permutes every predictor `R` times and predicts on all `n` rows, permutation importance has illustrative prediction cost `O(R * p * n * T * d)`, plus data copying/shuffling. This expression is conditional: the helper source has not been provided.

SHAP complexity cannot be determined from the wrapper alone. If the helper performs `Q` full-model evaluations on batches of size `b`, its prediction work is approximately `O(Q * b * T * d)`. `Q` and `b` must be determined from `calculateSHAP` before any quantitative claim is made.

## Memory complexity

- Dense double-precision input X: `8*n*p` bytes (excluding MATLAB array/object overhead and copies).
- Stored ensemble: approximately `O(T*s)` nodes plus per-node statistics and metadata. For fully grown trees, `s` may scale with `n`.
- SHAP output (if dense n-by-p doubles): `8*n*p` bytes, in addition to temporary buffers.
- Permutation helper may allocate one or more n-by-p copies; exact peak memory cannot be inferred without its code and runtime measurements.

### Example data-storage estimates (MiB)

| n | p | Input X | SHAP values (if n x p doubles) | Combined lower-level array storage |
|---:|---:|---:|---:|---:|
| 239 | 249 | 0.45 | 0.45 | 0.91 |
| 500 | 1,000 | 3.81 | 3.81 | 7.63 |
| 1,000 | 5,000 | 38.15 | 38.15 | 76.29 |
| 5,000 | 20,000 | 762.94 | 762.94 | 1,525.88 |

These are data-array sizes, **not** total or peak RAM measurements. The RF model, additional copies, and MATLAB Runtime are excluded.

## Implementation-specific findings

1. `app.number_of_decisionsTrees_input` is a direct and important computational scaling parameter.
2. The multivariate branch collapses outcomes to their rowwise mean (`mean(y,2)`); this is a single-target RF model.
3. SHAP is computed with training and explanation data both set to `X`. These are in-sample explanations; interpretability should not be confused with out-of-sample performance.
4. OOB predictions should not be described as independent held-out validation. Their statistical properties differ from a separate test set.
5. In the univariate branch with multiple outcomes, the two assignments to `app.Pred_Val` and `app.Orig_Val` are duplicated; they do not represent a second prediction computation of consequence.
6. The exact complexities of `calculateRFPermutationImportance`, `calculateSHAP`, and `oobQuantilePredict` remain unresolved until helper implementations and MATLAB settings are examined.

## Planned empirical validation

Record wall-clock runtime and peak process memory separately for ensemble training, OOB prediction, permutation importance, and SHAP. Repeat under documented hardware/software configurations and systematically vary n, p, T, and explanation sample counts. Do not use this theoretical report as evidence of measured cross-platform performance.
