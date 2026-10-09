# PANOMICs — CNN: Theoretical Computational Complexity

**Status:** Code-derived analytical assessment; not an empirical runtime or peak-memory benchmark.

## 1. Implementation reviewed

The supplied `CNN(app)` routine uses an 80:20 index-based train/test split; `sequenceInputLayer(p)`; three 1-D convolution layers with kernel widths `k1`, `k2`, `k3` and output channel counts `p`, `2p`, `p`; ReLU activations; fully connected layers of widths `d1`, `d2`, then a scalar regression output. Training uses Adam, CPU execution, user-selected epoch count `E`, and mini-batch size `b`. The workflow additionally computes permutation feature importance and SHAP on the test set for a newly trained model. In multivariate mode, targets are averaged across output columns before training.

**Crucial shape caveat:** `trainNetwork(inputTrain', outputTrain', layers, options)` uses numeric transposed matrices with a sequence input layer. The intended observation/sequence/time dimensions and MATLAB release-specific input interpretation must be verified before assigning a definitive sequence length or training complexity. The expressions below are conditional on a valid sequence representation with `N` independent training sequences, each of length `s`.

## 2. Parameter count

Let `p` be the number of input features, `k1`, `k2`, `k3` the three convolution kernel widths, and `d1`, `d2` the fully connected hidden widths. With convolution biases, the parameter counts are:

- Conv1, p input channels -> p output channels: `k1*p^2 + p`.
- Conv2, p -> 2p channels: `2*k2*p^2 + 2p`.
- Conv3, 2p -> p channels: `2*k3*p^2 + p`.
- FC1, p -> d1: `p*d1 + d1`.
- FC2, d1 -> d2: `d1*d2 + d2`.
- Output, d2 -> 1: `d2 + 1`.

Total trainable parameters:

`P = (k1 + 2*k2 + 2*k3)*p^2 + (4 + d1)*p + d1 + d1*d2 + 2*d2 + 1`.

This assumes the fully connected layers operate independently over sequence positions and do not flatten the sequence. If the MATLAB data layout or layer behavior differs, the count must be re-evaluated.

## 3. Time complexity

A representative forward-pass cost per sequence time step is

`O( (k1 + 2*k2 + 2*k3)*p^2 + p*d1 + d1*d2 + d2 )`.

For `N` training sequences, sequence length `s`, and `E` epochs, the approximate training-work scaling, including a constant-factor backpropagation overhead, is

`O( E*N*s*[(k1 + 2*k2 + 2*k3)*p^2 + p*d1 + d1*d2 + d2] )`.

The actual cost depends on input layout, convolution implementations, mini-batching, optimizer overhead, and available CPU resources. `MiniBatchSize` affects per-step overhead and activation storage; it does not simply multiply the total number of examples processed.

## 4. Memory complexity

- Dense input data: `O(n*p)` for `n` observations and `p` features, excluding copies and response matrices.
- Network parameters and Adam optimizer states: `O(P)` each; Adam typically maintains two moment arrays in addition to model parameters and gradients.
- Batch activations: approximately `O(b*s*(p + 2p + p + d1 + d2))`, with additional saved intermediates needed for backpropagation.
- SHAP output storage, if all feature attributions for `n_test` rows are retained: `O(n_test*p)`.

These are asymptotic storage models, **not measured MATLAB process peak RAM**. GPU memory is not relevant to this code path because `ExecutionEnvironment` is set to `cpu`.

## 5. Prediction and explainability overhead

For a single inference pass, replace `E*N` by the number of prediction sequences. The helper functions `calculateCNNPermutationImportance` and `calculateSHAP` can require numerous extra inference passes; their exact computational complexity cannot be determined without their definitions. For a conventional permutation method using `R` repetitions over `p` features, the prediction workload is roughly `R*p` evaluations over the selected test subset, plus baseline predictions. SHAP complexity is implementation-dependent.

## 6. Implementation-specific methodological notes

1. The training/test split uses contiguous rows, with no shuffling. This can be appropriate for temporally ordered data, but may bias evaluations when row ordering carries class, batch, or population structure.
2. The model has one output neuron. In multivariate mode, the code trains on `mean(outputTrain,2)` rather than jointly predicting all output variables.
3. The number of filters is tied to `p`; this can lead to quadratic parameter growth in high-dimensional omics applications.
4. There is no validation split or early-stopping rule shown in the supplied training options.
5. A reused network bypasses training and explainability calculations; prediction still runs. Therefore training and reuse modes must be reported separately.
6. The exact behavior of `predict(net,inputTest','MiniBatchSize',1)` and the returned output orientation should be checked with a small known dataset.
7. The values of `k1`, `k2`, `k3`, `d1`, `d2`, `E`, `b`, and the helper-function definitions are needed to produce implementation-specific numeric parameter counts or model operation curves.
