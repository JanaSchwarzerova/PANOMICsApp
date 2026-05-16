function results = runElasticNetRegression( ...
    X, ...
    y, ...
    alphaValue, ...
    cvFolds, ...
    holdoutRatio ...
)
%RUNELASTICNETREGRESSION Train and evaluate an Elastic Net regression model.
%
% This function performs:
%   1. Train/test split
%   2. Elastic Net regression using MATLAB's LASSO solver
%   3. Cross-validation for lambda optimization
%   4. Lambda selection using the 1-SE rule
%   5. Prediction on unseen test data
%
% NOTE:
%   MATLAB implements Elastic Net through the "lasso" function:
%
%       Alpha = 1   -> LASSO
%       Alpha = 0   -> Ridge
%       0 < Alpha < 1 -> Elastic Net
%
% INPUTS:
%   X             - Feature matrix [nSamples x nFeatures]
%   y             - Target/output vector [nSamples x 1]
%   alphaValue    - Elastic Net mixing parameter
%                   Recommended range: 0.1 - 0.9
%   cvFolds       - Number of cross-validation folds
%   holdoutRatio  - Test split ratio (default = 0.3)
%
% OUTPUT:
%   results - Structure containing:
%
%       .coefficients      -> Selected model coefficients
%       .intercept         -> Model intercept
%       .predictedValues   -> Predictions on test set
%       .originalValues    -> Ground truth values
%
%       .elasticNetWeights -> Full coefficient matrix
%       .fitInfo           -> MATLAB fit information
%
%       .trainIndices      -> Training sample indices
%       .testIndices       -> Test sample indices
%
%       .bestLambdaIndex   -> Selected lambda index
%       .bestLambdaValue   -> Selected lambda value
%
% EXAMPLE:
%   results = runElasticNetRegression( ...
%       X, ...
%       y, ...
%       0.5, ...
%       10, ...
%       0.3 ...
%   );

    %% Default parameters

    if nargin < 5
        holdoutRatio = 0.3;
    end

    %% Validate alpha parameter

    if alphaValue < 0 || alphaValue > 1
        error('alphaValue must be between 0 and 1.');
    end

    %% Create train/test split

    numSamples = length(y);

    cvPartition = cvpartition( ...
        numSamples, ...
        'HoldOut', ...
        holdoutRatio ...
    );

    trainIdx = training(cvPartition, 1);
    testIdx = ~trainIdx;

    XTrain = X(trainIdx, :);
    yTrain = y(trainIdx);

    XTest = X(testIdx, :);
    yTest = y(testIdx);

    %% Train Elastic Net model

    [elasticNetWeights, fitInfo] = lasso( ...
        XTrain, ...
        yTrain, ...
        'Alpha', alphaValue, ...
        'CV', cvFolds ...
    );

    %% Select optimal lambda using 1-SE rule

    bestLambdaIdx = fitInfo.Index1SE;

    selectedCoefficients = ...
        elasticNetWeights(:, bestLambdaIdx);

    interceptValue = ...
        fitInfo.Intercept(bestLambdaIdx);

    bestLambdaValue = ...
        fitInfo.Lambda(bestLambdaIdx);

    %% Predict on test dataset

    predictedValues = ...
        XTest * selectedCoefficients + interceptValue;

    %% Store results

    results = struct();

    results.coefficients = selectedCoefficients;
    results.intercept = interceptValue;

    results.predictedValues = predictedValues;
    results.originalValues = yTest;

    results.elasticNetWeights = elasticNetWeights;
    results.fitInfo = fitInfo;

    results.trainIndices = trainIdx;
    results.testIndices = testIdx;

    results.bestLambdaIndex = bestLambdaIdx;
    results.bestLambdaValue = bestLambdaValue;

end