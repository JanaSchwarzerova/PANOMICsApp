# PANOMICs — Support Vector Regression (SVR): Implementation-Specific Computational Complexity

**Status:** Theoretical, code-based assessment. **No measured runtime or peak RAM is reported.**

## 1. Scope and implementation

This assessment is based on the supplied `SVR(app)` MATLAB App Designer function. Both the univariate and multivariate branches fit five SVR models (`fitrsvm`) on manually constructed training/test splits. Each fold includes prediction, permutation feature importance (`app.calculatePermutationImportance`), and SHAP (`app.calculateSHAP`). In the multivariate case, the target is reduced to `mean(Y,2)`; this is **not** multi-output SVR. Only the model from the last fold is retained as `MdlLin`.

`fitrsvm(X,Y)` is invoked without an explicit kernel, solver, standardization, or optimization configuration. For MATLAB's standard `fitrsvm` defaults, the kernel is linear; the exact optimization workload remains implementation- and data-dependent. This is distinct from an assumed RBF/kernel-matrix SVR benchmark.

## 2. Notation

- `n`: number of samples; `p`: number of predictors (SNPs/metabolites).
- `K=5`: number of model-fitting passes.
- `n_tr ≈ 0.8n`, `n_te ≈ 0.2n`: intended training/test sizes per pass; actual code deviates at fold boundaries.
- `S`: number of support vectors (for support-vector prediction representation).
- `R`: effective number of feature-permutation repetitions, determined by the helper implementation.
- `H`: effective number of model evaluations made by the SHAP helper.
- `C_fit(n_tr,p,solver,data)`: cost of one `fitrsvm` training run.
- `C_pred(n_te,p,S,kernel)`: cost of predicting one test fold.

## 3. Time complexity

The training cost of a support vector regressor is not described reliably by a single universal big-O bound without the solver, kernel, stopping tolerances, and data distribution. The five model fits contribute

`T_train = Σ_{k=1..5} C_fit(n_tr,k, p, solver, data)`.

For a linear predictor, scoring `n_te` samples is approximately `O(n_te × p)` after forming a linear weight representation. For a support-vector expansion, prediction can require `O(n_te × S × p)` kernel-related work. Which representation is used depends on the fitted model and MATLAB's implementation.

The full workflow additionally includes:

- Five ordinary prediction calls.
- Permutation importance: approximately `O(K × R × p × C_pred(n_te,...))` if the helper permutes every feature `R` times and evaluates the entire test fold each time; actual cost is **unknown** until the helper is inspected.
- SHAP: approximately `O(K × H × C_model_eval)` with `H` and the evaluation batch size determined by `app.calculateSHAP`; this is a structural expression, **not** a measured or guaranteed upper bound.
- Fold slicing and SHAP aggregation: typically at least `O(K × n × p)` copying/processing in this MATLAB implementation.

**Illustrative work proxy, not runtime:** for a fixed number of linear-model iterations, the amount of data processed per training pass scales roughly with `n_tr × p`; five passes multiply that workload. This proxy omits solver-specific convergence and explainability overhead and must not be used to compare absolute speed with other algorithms.

## 4. Memory complexity

- Dense input matrix of doubles: `8 × n × p` bytes (`n × p × 8 / 2^20` MiB).
- Training/test subsets: additional data copies may be materialized by MATLAB indexing.
- `featureImportanceAll`, `shapImportanceAll`, `MeanSHAPAll`: each `p × 5` doubles, together `120p` bytes, excluding metadata.
- Per-fold `SHAPValuesFold`: if dense, approximately `8 × n_te × p` bytes, plus helper-specific temporary arrays.
- Model state, solver workspaces, and permutation/SHAP temporaries: unknown without measurements and helper code.

**These quantities are storage estimates, not peak process RAM.**

| Samples (`n`) | Features (`p`) | One dense input matrix (MiB) |
|---:|---:|---:|
| 239 | 249 | 0.45 |
| 500 | 1,000 | 3.81 |
| 1,000 | 5,000 | 38.15 |
| 5,000 | 20,000 | 762.94 |

## 5. Implementation findings relevant to reproducibility

1. **Non-disjoint folds.** For `i>1`, the test interval begins at `floor(n/5)*(i-1)` and the training set includes the same boundary index (`...*(i-1)-1` then `...*i:end`); the last test index is also included in training. Thus the manually defined splits leak test samples into training. The final remainder when `n` is not divisible by five is not tested in the intended five-fold coverage.
2. **SHAP mean not populated.** `MeanSHAPAll` is initialized to zero and never updated, so saved `MeanSHAP` is zero irrespective of the computed SHAP values.
3. **Model caching / stale outputs.** If `app.SVR_univar_mat` or `app.SVR_multivar_mat` is already populated, the corresponding computation is skipped. Since `ypred11` and `testM1` are initialized empty at the beginning, `app.Pred_Val` and `app.Orig_Val` can become empty on a subsequent call.
4. **Multivariate interpretation.** The target `mean(Y,2)` collapses multiple outcomes into one mean response. Results must not be described as separate predictions for each phenotype.
5. **Last-fold model only.** `app.SVR_*_mat.MdlLin = MdlLin` retains the last fold's model, not a refit using all training observations.
6. **Unknown helper costs.** The functions `app.calculatePermutationImportance` and `app.calculateSHAP` were not supplied, so exact counts of prediction calls and temporary memory cannot be derived.

## 6. Suggested benchmark instrumentation

Measure independently (a) SVR training, (b) ordinary prediction, (c) permutation importance, and (d) SHAP. Log dataset dimensions, MATLAB version, hardware, solver/kernel settings, support-vector count, and error status. Use an operating-system process monitor for **peak resident memory**, as memory snapshots at the start and end do not establish peak RAM. Report desktop and web deployment separately. Do not publish illustrative complexity proxies as empirical seconds or RAM measurements.

## 7. Suggested manuscript wording

> We additionally documented the implementation-specific computational structure of the SVR workflow, comprising five model-fitting passes, held-out prediction, permutation-based feature importance, and SHAP-based interpretation. The theoretical analysis identifies training, repeated model evaluations, and explainability procedures as distinct contributors to computational cost. These estimates characterize expected workload scaling but do not substitute for empirical runtime and peak-memory benchmarking of the desktop and web implementations.

## 8. Next step

Inspect `app.calculatePermutationImportance` and `app.calculateSHAP`, then perform measured benchmarks on leakage-free, disjoint folds. A suitable split generator is `cvpartition(n,'KFold',5)` with `training(c,i)` and `test(c,i)`.
