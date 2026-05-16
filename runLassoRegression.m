function results = runLassoRegression(X, y, alphaValue, cvFolds, holdoutRatio)
%RUNLASSOREGRESSION Train and evaluate a LASSO regression model.
%
% This function performs:
%   1. Train/test split
%   2. LASSO regression with cross-validation
%   3. Selection of the optimal lambda using the 1-SE rule
%   4. Prediction on test data
%
% INPUTS:
%   X             - Input feature matrix [nSamples x nFeatures]
%   y             - Target/output vector [nSamples x 1]
%   alphaValue    - LASSO alpha parameter
%                   (1 = LASSO, 0 = Ridge, 0-1 = Elastic Net)
%   cvFolds       - Number of folds for cross-validation
%   holdoutRatio  - Ratio of test samples (e.g. 0.3 = 30 % test data)
%
% OUTPUT:
%   results - Structure containing:
%       .coefficients      -> Selected model coefficients
%       .intercept         -> Model intercept
%       .predictedValues   -> Predictions for test set
%       .originalValues    -> Ground truth test values
%       .lassoWeights      -> Full LASSO coefficient matrix
%       .fitInfo           -> MATLAB LASSO fit information
%       .trainIndices      -> Logical indices for training data
%       .testIndices       -> Logical indices for testing data
%
% EXAMPLE:
%   results = runLassoRegression(X, y, 1, 10, 0.3);

    %% Input validation

    if nargin < 5
        holdoutRatio = 0.3;
    end

    %% Create train/test split

    numSamples = length(y);

    cvPartition = cvpartition(numSamples, 'HoldOut', holdoutRatio);

    trainIdx = training(cvPartition, 1);
    testIdx = ~trainIdx;

    XTrain = X(trainIdx, :);
    yTrain = y(trainIdx);

    XTest = X(testIdx, :);
    yTest = y(testIdx);

    %% Train LASSO model

    [lassoWeights, fitInfo] = lasso( ...
        XTrain, ...
        yTrain, ...
        'Alpha', alphaValue, ...
        'CV', cvFolds ...
    );

    %% Select optimal lambda using 1-SE rule

    bestLambdaIdx = fitInfo.Index1SE;

    selectedCoefficients = lassoWeights(:, bestLambdaIdx);

    interceptValue = fitInfo.Intercept(bestLambdaIdx);

    %% Predict on test data

    predictedValues = XTest * selectedCoefficients + interceptValue;

    %% Store results

    results = struct();

    results.coefficients = selectedCoefficients;
    results.intercept = interceptValue;

    results.predictedValues = predictedValues;
    results.originalValues = yTest;

    results.lassoWeights = lassoWeights;
    results.fitInfo = fitInfo;

    results.trainIndices = trainIdx;
    results.testIndices = testIdx;

end