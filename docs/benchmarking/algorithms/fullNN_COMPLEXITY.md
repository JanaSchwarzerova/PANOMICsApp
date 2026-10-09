# PANOMICs — Fully Connected Neural Network (fullNN): Computational Complexity

**Status:** Code-derived theoretical assessment; **no runtime or peak-memory measurements were performed**.

## Actual MATLAB implementation

`fullyconnected_NN(app)` uses a contiguous 80:20 row split, `featureInputLayer(p)`, three hidden fully connected layers of widths `h1`, `h2`, `h3` with ReLU, and one regression output. Training uses Adam **on CPU** with `Shuffle='never'`; epochs are controlled by **`app.max_epoches_CNN_input`**, and gradient clipping by **`app.gradientTreshold_CNN_input`** (both CNN-named properties). New-model paths compute permutation feature importance and SHAP on held-out rows. Cached-model paths skip training and attribution and only predict. Multivariate mode fits the **mean of response columns**, not a multiple-output network.

## Exact parameter count for the specified architecture

For `p` input predictors and hidden widths `h1,h2,h3`, including biases:

`P = (p+1)h1 + (h1+1)h2 + (h2+1)h3 + (h3+1)`.

Equivalently `P = p*h1 + h1*h2 + h2*h3 + h1+h2+2*h3+1`. At fixed hidden widths, parameter storage grows **linearly in p**, unlike the provided CNN architecture whose convolutional channel counts depend on p.

## Time complexity (operation scaling, not seconds)

For `n_tr=floor(0.8n)` training observations and `E` epochs, the leading dense matrix multiplication work is approximately

`W_train = O(E*n_tr*(p*h1+h1*h2+h2*h3+h3))`.

Backpropagation and Adam add constant factors and overhead. Prediction over `n_te` rows is `O(n_te*(p*h1+h1*h2+h2*h3+h3))`. Permutation importance and SHAP each invoke additional predictions; their exact number and memory requirements cannot be determined without the helper implementations. `MiniBatchSize=1` in inference can add substantial per-call overhead.

## Memory complexity

- Dense input storage (double): `8*n*p` bytes; train/test slices and attribution arrays may create further copies.
- Model weights (single precision illustrative): `4P` bytes; gradient plus two Adam moment arrays bring a **minimum four-array accounting** to approximately `16P` bytes, before other optimizer/activation/runtime buffers. MATLAB's actual precision and storage must be verified.
- Hidden activations for one mini-batch of `b` rows: at least `O(b*(p+h1+h2+h3))` elements, plus backpropagation workspace.
- Stored SHAP matrix: `O(n_te*p)` elements.

A practical symbolic estimate is `O(np+P+b(p+h1+h2+h3)+n_te*p)` plus helper-function workspaces. This is **not peak resident RAM**.

## Implementation and scientific review

1. CPU is hard-coded. No desktop-versus-web performance difference can be inferred without measuring both deployments.
2. The network is a single-response regressor; multivariate output is reduced to `mean(outputTrain,2)`.
3. Epoch and gradient-threshold controls refer to **CNN** app properties, not fullNN-specific properties. Verify whether this is intended.
4. The 80:20 split preserves row order. No separate validation or early stopping is configured here.
5. The code uses `size(outputTrain,2)==1` for choosing original targets; this is consistent with a single-response target but depends on response orientation.
6. The cached-model path does not recompute SHAP or permutation importance. Fresh and cached workflows are not comparable as equivalent runtime jobs.
7. SHAP output size, predictor shape, and helper prediction-call count must be verified.

