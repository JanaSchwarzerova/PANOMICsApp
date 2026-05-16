function results = runSVRRegression(X, y, cvFolds)
%RUNSVRREGRESSION Train and evaluate Support Vector Regression model.
%
% This implementation uses k-fold cross-validation (default: 5 folds)
% and MATLAB's fitrsvm() for regression.
%
% PIPELINE:
%   1. Manual k-fold split (sequential blocks)
%   2. Train SVR model on training data
%   3. Predict on test fold
%   4. Aggregate predictions
%
% NOTE:
%   - If Y is multivariate, target is reduced using mean(Y,2)
%
% INPUTS:
%   X        - Feature matrix [nSamples x nFeatures]
%   y        - Target vector or matrix
%   cvFolds  - Number of folds (default = 5)
%
% OUTPUT:
%   results struct:
%       .predictedValues
%       .originalValues
%       .model
%       .foldModels
%       .foldPredictions

    %% Defaults
    if nargin < 3
        cvFolds = 5;
    end

    n = size(X,1);
    foldSize = floor(n / cvFolds);

    yPredAll = [];
    yTrueAll = [];
    foldModels = cell(cvFolds,1);

    %% Cross-validation loop
    for i = 1:cvFolds

        %% Define test indices (same logic as original code)
        if i == 1
            testIdx = 1:foldSize*i;
        else
            testIdx = (foldSize*(i-1)) : (foldSize*i);
        end

        trainIdx = setdiff(1:n, testIdx);

        XTrain = X(trainIdx,:);
        XTest  = X(testIdx,:);

        yTrain = y(trainIdx,:);
        yTest  = y(testIdx,:);

        %% Handle multivariate target
        if size(yTrain,2) > 1
            yTrainModel = mean(yTrain,2);
            yTestModel  = mean(yTest,2);
        else
            yTrainModel = yTrain;
            yTestModel  = yTest;
        end

        %% Train SVR model
        svrModel = fitrsvm(XTrain, yTrainModel);

        %% Predict
        yPred = predict(svrModel, XTest);

        %% Store results
        yPredAll = [yPredAll; yPred];
        yTrueAll = [yTrueAll; yTestModel];

        foldModels{i} = svrModel;
    end

    %% Output structure
    results = struct();

    results.predictedValues = yPredAll;
    results.originalValues = yTrueAll;

    results.foldModels = foldModels;

    %% Representative model (last fold)
    results.model = foldModels{end};

end