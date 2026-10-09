# PANOMICs – Partial Least Squares Regression (PLS)

**Status:** Implementation-informed theoretical complexity assessment; **not** an empirical runtime or peak-memory benchmark.

## Implementation reviewed

The supplied `PLS(app)` function has two pathways, controlled by `app.univar` and `app.multivar`. When the relevant cached result is empty, each pathway executes five manually constructed train/test partitions. For each partition, the function:

1. Fits `[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(trainG,trainM)` without specifying `ncomp`.
2. Predicts test outcomes with `[ones(size(testG,1),1) testG] * beta`.
3. Calls `app.calculatePermutationImportance` on held-out data.
4. Calls `app.calculateSHAP` on the fitted linear prediction function.
5. Stores fold-level permutation and absolute-SHAP importance and aggregates these over five iterations.

In the multivariate pathway, PLS predicts all columns of `trainM`; permutation importance and SHAP instead explain the **row-wise mean** of the predicted outcomes. The final PLS model and scores stored in `app.PLS_*_mat` correspond only to the last partition, while feature-importance summaries aggregate all partitions.

## Symbols and assumptions

- `n`: number of input observations; `p`: number of predictor columns; `q`: number of response columns.
- `K = 5`: number of attempted validation partitions.
- `n_tr ≈ 0.8 n`, `n_te ≈ 0.2 n`: nominal sizes, assuming the fold boundaries are corrected.
- `a`: number of PLS latent components, which is **not explicitly set** in the supplied code. Its actual value depends on MATLAB defaults and the matrix dimensions.
- `I`: effective iterative work per component, if an iterative decomposition is used.
- `R`: permutation repeats per feature (not provided).
- `S`: number of prediction-model evaluations required by the SHAP helper (not provided).

## Time complexity

A useful **illustrative computational-work model** for iterative multivariate PLS fitting is

`W_fit = O(K * I * n_tr * p * q * a)`.

This is **not a verified MATLAB `plsregress` solver bound**. Exact costs depend on its internal decomposition, rank, default component count, data dimensions, and numerical convergence. For a single response (`q = 1`), the expression simplifies to `O(K * I * n_tr * p * a)`.

A fitted linear PLS prediction for a test fold costs `O(n_te * p * q)`; all five folds cost `O(K * n_te * p * q)`. Permutation importance and SHAP add repeated model evaluations. If the permutation helper makes `R` full-fold predictions for each of `p` features, its prediction work is approximately `O(K * R * n_te * p^2 * q)` in the univariate case (plus permutations and metric evaluation). The actual helper code is required to confirm `R`, batching, and scoring details. SHAP costs cannot be quantified without the `calculateSHAP` implementation and its background/evaluation sampling strategy.

The total workflow can therefore be expressed as:

`T_total = T_PLS_fitting + T_prediction + T_permutation + T_SHAP + T_data_handling`.

## Space complexity

- Dense input predictor matrix: `O(n*p)` doubles; dense response matrix: `O(n*q)` doubles.
- Regression coefficients `beta`: `O((p+1)*q)` doubles.
- PLS loadings/scores: typically include `O(p*a + q*a + n_tr*a)`-sized arrays, with exact shapes depending on outputs.
- Five-fold importance summaries: `O(K*p)` doubles per stored importance matrix.
- SHAP values for a held-out fold: potentially `O(n_te*p)` doubles if materialized.
- Additional solver workspaces, train/test copies, and SHAP background data are implementation-dependent. These estimates do **not** represent MATLAB process peak RAM.

For illustration, a dense `double` predictor matrix alone requires `8*n*p` bytes (before copies):

| Samples (`n`) | Predictors (`p`) | Input matrix only (MiB) |
|---:|---:|---:|
| 239 | 249 | 0.45 |
| 500 | 1,000 | 3.81 |
| 1,000 | 5,000 | 38.15 |
| 5,000 | 20,000 | 762.94 |

## Implementation-specific cautions

1. **Cross-validation leakage:** For folds 2–5, the test interval begins at `floor(n/5)*(i-1)` while the training indices include that same boundary observation. The observation is present in both sets. The last remainder observations are not assigned to a held-out fold when `n` is not divisible by five. This is not valid disjoint five-fold CV. Use a `cvpartition` object and its `training`/`test` masks.
2. **Signed SHAP:** `MeanSHAPAll` is initialized to zeros and never updated. Consequently, the reported `MeanSHAP` is zero regardless of the actual signed attributions. Within each fold, compute `mean(SHAPValuesFold,1)'` and store it in the corresponding column.
3. **Number of components:** `ncomp` is not specified, so the analysis must not claim a fixed component count or fixed convergence cost. Explicitly setting and selecting `ncomp` using leakage-safe inner validation would improve reproducibility.
4. **Multivariate attribution:** `beta` yields a multivariate prediction, but the permutation and SHAP callbacks explain its mean across outputs. This should be described as a mean-response explanation, not feature importance for every response separately.
5. **Model retention:** The saved `beta`, `XL`, `XS`, and other fit objects are from the final fold only, not a model refitted on all training data.
6. **Two branches:** Both univariate and multivariate pathways can run if both mode flags are enabled; their work should not be counted as a single five-fit workflow in that case.

## Reproducibility requirements for future empirical benchmarks

Record MATLAB release, operating system, CPU, RAM, PLS component count, input dimensions, five disjoint validation folds, permutation repeat count, SHAP configuration, elapsed time, and process peak memory. Report fitting-only and end-to-end workflow measurements separately.
