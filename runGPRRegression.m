function results = runGPRRegression(X, y, cvFolds)
%RUNGPRREGRESSION Train and evaluate Gaussian Process Regression model.
%
% This implementation uses k-fold cross-validation (default: 5-fold)
% and MATLAB's fitrgp() for Gaussian Process Regression.
%
% PIPELINE:
%   1. K-fold split (manual, sequential blocks)
%   2. Train GPR model on training folds
%   3. Predict on validation fold
%   4. Aggregate predictions
%
% NOTE:
%   - If Y is multivariate, target is reduced using mean(Y,2)
%
% INPUTS:
%   X        - Feature matrix [nSamples x nFeatures]
%   y        - Target vector or matrix [nSamples x 1 or nTargets]
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

        %% Define test indices (sequential split like original code)

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

        %% Reduce to univariate if needed
        if size(yTrain,2) > 1
            yTrainModel = mean(yTrain,2);
            yTestModel  = mean(yTest,2);
        else
            yTrainModel = yTrain;
            yTestModel  = yTest;
        end

        %% Train GPR model
        gprModel = fitrgp(XTrain, yTrainModel);

        %% Predict
        yPred = predict(gprModel, XTest);

        %% Store results
        yPredAll = [yPredAll; yPred];
        yTrueAll = [yTrueAll; yTestModel];

        foldModels{i} = gprModel;

    end

    %% Output struct
    results = struct();

    results.predictedValues = yPredAll;
    results.originalValues = yTrueAll;

    results.foldModels = foldModels;

    %% Store last model as representative (optional)
    results.model = foldModels{end};

end