function results = runRidgeRegression( ...
    X, ...
    y, ...
    cvFolds, ...
    holdoutRatio ...
)
%RUNRIDGEREGRESSION Train and evaluate Ridge Regression model.
%
% NOTE:
% MATLAB Ridge Regression is implemented via:
%   lasso(X, y, 'Alpha', 0)
%
% Ridge shrinks coefficients but does NOT enforce sparsity.
%
% PIPELINE:
%   1. Train/test split
%   2. Ridge regression (alpha = 0)
%   3. Cross-validation over lambda
%   4. Selection using 1-SE rule
%   5. Prediction on test set
%
% INPUTS:
%   X             - Feature matrix [nSamples x nFeatures]
%   y             - Target vector [nSamples x 1]
%   cvFolds       - Cross-validation folds
%   holdoutRatio  - Test split ratio (default 0.3)
%
% OUTPUT:
%   results struct:
%       .coefficients
%       .intercept
%       .predictedValues
%       .originalValues
%       .ridgeWeights
%       .fitInfo
%       .trainIndices
%       .testIndices
%       .bestLambdaIndex
%       .bestLambdaValue

    %% Defaults
    if nargin < 4
        holdoutRatio = 0.3;
    end

    %% Split data
    n = length(y);

    cv = cvpartition(n, 'HoldOut', holdoutRatio);

    trainIdx = training(cv, 1);
    testIdx  = ~trainIdx;

    XTrain = X(trainIdx, :);
    yTrain = y(trainIdx);

    XTest = X(testIdx, :);
    yTest = y(testIdx);

    %% Ridge regression (via lasso with alpha = 0)
    [ridgeWeights, fitInfo] = lasso( ...
        XTrain, ...
        yTrain, ...
        'Alpha', 0, ...
        'CV', cvFolds ...
    );

    %% Lambda selection (1-SE rule)
    bestIdx = fitInfo.Index1SE;

    coef = ridgeWeights(:, bestIdx);
    intercept = fitInfo.Intercept(bestIdx);

    bestLambda = fitInfo.Lambda(bestIdx);

    %% Prediction
    yPred = XTest * coef + intercept;

    %% Output struct
    results = struct();

    results.coefficients = coef;
    results.intercept = intercept;

    results.predictedValues = yPred;
    results.originalValues = yTest;

    results.ridgeWeights = ridgeWeights;
    results.fitInfo = fitInfo;

    results.trainIndices = trainIdx;
    results.testIndices = testIdx;

    results.bestLambdaIndex = bestIdx;
    results.bestLambdaValue = bestLambda;

end