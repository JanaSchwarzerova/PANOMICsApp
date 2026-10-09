# PANOMICs — Ridge Regression (RR): Implementation-Specific Complexity Analysis

**Status:** Analytical workload model based on supplied MATLAB App Designer source code. **Not an empirical runtime or peak-RAM benchmark.**

## 1. Implementation inspected

The `RR(app)` routine reads `app.x_input` and `app.y_output`, generates a random 70% training / 30% test partition using `cvpartition(n,'HoldOut',0.3)`, and on a fresh-model branch calls:

```matlab
[B,FitInfo] = lasso(XTrain,yTrain,'Alpha',app.RR_alfa,'CV',app.RR_cv);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
```

It predicts with `XTest*coef + coef0` and calculates coefficient-based attribution via `(XTest - mean(XTrain,1)).*coef'`. It stores the regularization-path coefficients `B` and cross-validation metadata `FitInfo`. The routine has separate univariate and multivariate branches, with optional reuse of stored coefficients rather than refitting.

**Important model-definition qualification:** MATLAB's `lasso` uses elastic-net mixing parameter `Alpha` in `(0,1]`; positive `Alpha` does not implement exact pure L2 ridge regression. A small positive value approximates ridge-like penalization. The exact value of `app.RR_alfa` is not provided, so this report describes the *implemented `lasso` solver*, not a conventional closed-form ridge solver. Do not label the fitted model as exact ridge regression without checking the estimator definition.

## 2. Variables and assumptions

| Symbol | Meaning |
|---|---|
| `n` | Number of input samples |
| `p` | Number of input features (SNPs / metabolites) |
| `n_tr` | Training samples, approximately `0.7n` |
| `n_te` | Testing samples, approximately `0.3n` |
| `K` | Number of CV folds, `app.RR_cv` |
| `L` | Number of lambda values evaluated by `lasso` (MATLAB defaults/configuration may vary) |
| `I` | Effective optimization passes per lambda/fold, data- and tolerance-dependent |

Assume dense `double` input for illustrative memory estimates. Solver convergence, feature correlations, sparsity, lambda-path length, and MATLAB internals affect actual runtime.

## 3. Time-complexity workload model

For coordinate-descent-style elastic-net/lasso fitting on dense inputs, a useful *illustrative* work proxy is:

\[
W_{\mathrm{fit}} \propto (K+1)\,L\,I\,n_{\mathrm{tr}}\,p.
\]

`K` represents CV training fits and `+1` an illustrative final fit on all training observations. This is not an exact accounting of MATLAB's internal operations, which may use screening, warm starts, early stopping, and other optimizations.

- Partitioning and extraction: typically proportional to the data copied, `O(np)`.
- Predictions: `O(n_te p)`.
- Mean-reference coefficient attribution and SHAP-value matrix: `O((n_tr+n_te)p) = O(np)`.
- Feature attribution summaries: `O(n_te p)`.
- Cached-model path: **no fresh model fitting**, but new holdout selection, prediction and attribution remain `O(np)`.

**Critical comparability note:** LASSO in the previously supplied code invokes the `lasso` solver twice on its fresh univariate path, whereas this RR implementation invokes it once. This implementation-level distinction affects work even if the algorithms have similar underlying optimization families.

## 4. Memory-complexity model

- Dense input matrix: `8np` bytes (excluding MATLAB array overhead).
- Explicit training/test subsets may add approximately another `8np` bytes if copied.
- Coefficient path `B`: approximately `8pL` bytes for dense representation.
- Test attribution matrix `SHAPValues`: approximately `8n_te p` bytes.
- Additional solver workspace, cross-validation folds, intermediate matrices, stored results, and MATLAB runtime overhead are **not** bounded by these figures.

Hence a simplified *data-storage* model is `O(np + pL)`; it **does not estimate measured peak resident memory**.

### Dense input storage

| Samples `n` | Features `p` | Input matrix, MiB (8np / 2^20) |
|---:|---:|---:|
| 239 | 249 | 0.45 |
| 500 | 1,000 | 3.81 |
| 1,000 | 5,000 | 38.15 |
| 5,000 | 20,000 | 762.94 |

## 5. Implementation-specific observations and validation risks

1. **Model identity:** `lasso(...,'Alpha',app.RR_alfa)` is not an exact ridge implementation for any valid positive `Alpha`; document `app.RR_alfa` and consider a true ridge estimator if that is the intended model.
2. **Reuse and resampling:** cached coefficients may be applied to a *new* randomly generated train/test split, potentially evaluating on samples used to fit the original cached model. Reuse therefore must not be treated as independent out-of-sample evaluation without preserving original split IDs.
3. **SHAP naming:** the expression `(XTest-XReference).*coef'` is an additive coefficient-based attribution relative to a mean baseline. It is not a generic model-agnostic SHAP estimation routine.
4. **Potential output-field bug:** in the cached multivariate branch, SHAP results are written to `app.RR_univar_mat` instead of `app.RR_multivar_mat`.
5. **Cross-validation:** the code performs CV for lambda selection within the 70% training partition, but preprocessing before this routine may still introduce leakage. No nested outer cross-validation is evident in the supplied routine.
6. **Mode handling:** both univariate and multivariate branches use `lasso` with vector-valued response; the code alone does not establish how multi-response targets are represented elsewhere in the application.

## 6. Interpretation for the PANOMICs scalability report

This code-based analysis supports a qualitative comparison of computational growth with dataset dimensions and selected hyperparameters. It cannot establish execution time in seconds, peak RAM in GiB, server resource quotas, desktop-versus-web differences, or failure thresholds. Those require actual instrumented runs of the packaged desktop app and web deployment.

**Reporting language:** "We provide implementation-specific analytical complexity estimates for the RR-labelled routine, with empirical runtime and peak-memory measurements reserved for subsequent validation."
