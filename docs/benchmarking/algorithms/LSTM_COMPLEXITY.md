# PANOMICs — LSTM Computational Complexity Analysis

**Status:** Theoretical, implementation-informed analysis; no MATLAB runtime or peak-memory measurements have been performed.

## Implementation inspected

The supplied `LSTM(app)` function uses a deterministic first-80% training / last-20% testing split, a CPU-only Adam optimizer, `MaxEpochs=app.max_epoches_LSTM_input`, `MiniBatchSize=app.miniBatchSize_LSTM`, gradient clipping, and `Shuffle='never'`. Its architecture is `sequenceInputLayer(p) → lstmLayer(h,OutputMode='sequence') → fullyConnectedLayer(d) → dropoutLayer → fullyConnectedLayer(1) → regressionLayer`. The univariate branch uses the target directly, while the multivariate branch reduces multiple target columns to their row-wise mean. Model training occurs only when the corresponding stored network is empty. Newly trained models additionally run permutation feature importance and SHAP; the function always makes final test predictions.

## Symbols

- `n`: total number of observations; `n_train=floor(0.8n)`, `n_test=n-n_train` **if the input layout indeed represents independent observations**.
- `p`: input features (`size(app.x_input,2)`).
- `h`: hidden units (`app.lstmlayer_LSTM_1`).
- `d`: width of the intermediate fully connected layer (`app.fullyconnectedlayer_LSTM_2`).
- `s`: sequence length (must be verified against the actual MATLAB input layout).
- `E`: maximum epochs; `b`: mini-batch size.
- `P`: number of learnable parameters.

## Parameter count

Ignoring nontrainable layer state, a standard LSTM layer has `4h(p+h+1)` trainable weights/biases. The subsequent dense layers have `d(h+1)` and `(d+1)` parameters, respectively. Hence

`P = 4h(p+h+1) + d(h+1) + (d+1)`.

The dropout layer has no learnable weights. Actual MATLAB layer parameter conventions should be verified against the trained network.

## Time complexity

A conventional forward/backward training-work model is

`W_train = O(E * n_train * s * [4h(p+h) + hd + d])`,

up to implementation-specific backward-pass and optimizer constants. This assumes `n_train` independent sequences of length `s`, which the provided matrix layout does **not** establish. If MATLAB interprets the transposed matrix as one sequence, the number and lengths of sequences—and hence scaling—are different. `b` affects optimizer update counts, temporary activations, and hardware efficiency, but does not simply divide total arithmetic work by `b`.

Inference has an analogous per-sequence forward-pass cost without backpropagation. Full workflow cost is `W_train + W_predict + W_permutation + W_SHAP`; the latter two cannot be quantified until the corresponding helper functions, their repetitions/background sizes, and predictor-call counts are inspected.

## Memory complexity

Parameter and Adam optimizer state scale as `O(P)`; Adam typically maintains two additional moment arrays per trainable parameter. During training, stored recurrent activations scale approximately as `O(b*s*h)` plus input and dense-layer activations, with implementation-dependent overhead. Data storage is `O(np)` for a dense double input matrix. **These are scaling models, not measured peak RAM.**

## Implementation and validity checks

1. **Input semantics:** `trainNetwork(inputTrain',outputTrain',layers,options)` passes a numeric matrix of dimensions `p × n_train`. Verify how the installed MATLAB version interprets the observation and time dimensions for `sequenceInputLayer` and whether this matches the intended supervised task. `OutputMode='sequence'` implies sequence-wise responses; the data may be treated as a time series rather than independent samples.
2. **Target assignment:** `if size(outputTrain,1) == 1` is generally false after an 80/20 split with multiple training rows. Consequently `app.Orig_Val = mean(outputTest,2)` may be applied even to a univariate response (numerically unchanged when it is a single column). It is not a reliable mode check; use `size(outputTest,2)==1` or `app.univar`/`app.multivar`.
3. **SHAP predictor shape:** `reshape(predict(net,Xnew'),[],1)` may flatten multiple time steps or responses into one vector. Verify one prediction per explained observation, otherwise the SHAP function's expected dimensions may be violated.
4. **Train/test split:** The first 80% versus last 20% is deterministic. It may be appropriate for temporal prediction, but should not be presented as randomized holdout validation. Check preprocessing leakage separately.
5. **No validation set:** No validation data are supplied to `trainingOptions`; there is no validation-based early stopping in the shown code.
6. **CPU:** `ExecutionEnvironment='cpu'` is explicit; no GPU acceleration is enabled in the active branch.
7. **Reuse branch:** When the model is reused, the function skips retraining and the new permutation/SHAP computations, so fresh and cached runs have materially different costs.
8. **Multivariate target:** `mean(outputTrain,2)` collapses multiple outcomes into one target; this is not multi-output LSTM regression.


## Benchmarking status

No measured CPU execution times, observed peak RAM, server quotas, or experimentally established maximum dataset sizes are claimed here. The theoretical model should be revisited after verifying the MATLAB sequence-input layout and the helper-function implementations.
