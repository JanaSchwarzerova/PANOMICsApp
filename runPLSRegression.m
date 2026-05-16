function results = runPLSRegression(X, y, cvFolds, nComponents)
%RUNPLSREGRESSION Train and evaluate Partial Least Squares Regression.
%
% This function performs:
%   1. Manual k-fold cross-validation
%   2. PLS model training via plsregress
%   3. Prediction using regression coefficients
%   4. Aggregation of fold predictions
%
% NOTE:
%   - Multivariate Y is reduced using mean(Y,2)
%   - Model uses linear regression on latent components
%
% INPUTS:
%   X           - Feature matrix [nSamples x nFeatures]
%   y           - Target vector or matrix
%   cvFolds     - Number of folds (default = 5)
%   nComponents - Number of latent components (optional)
%
% OUTPUT:
%   results struct:
%       .predictedValues
%       .originalValues
%       .beta
%       .plsModel
%       .foldModels

    %% Defaults
    if nargin < 3
        cvFolds = 5;
    end

    if nargin < 4
        nComponents = min(size(X,2), 10);
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

        %% Reduce multivariate target
        if size(yTrain,2) > 1
            yTrainModel = mean(yTrain,2);
            yTestModel  = mean(yTest,2);
        else
            yTrainModel = yTrain;
            yTestModel  = yTest;
        end

        %% Train PLS model
        [~,~,~,~,beta,~,~,stats] = plsregress( ...
            XTrain, ...
            yTrainModel, ...
            nComponents ...
        );

        %% Predict
        yPred = [ones(size(XTest,1),1) XTest] * beta;

        %% Store results
        yPredAll = [yPredAll; yPred];
        yTrueAll = [yTrueAll; yTestModel];

        foldModels{i} = struct( ...
            'beta', beta, ...
            'stats', stats ...
        );

    end

    %% Output struct
    results = struct();

    results.predictedValues = yPredAll;
    results.originalValues = yTrueAll;

    results.foldModels = foldModels;

    %% Representative model (last fold)
    results.beta = foldModels{end}.beta;

    results.plsModel = foldModels{end};

end