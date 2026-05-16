function results = runRandomForestRegression(X, y, numTrees)
%RUNRANDOMFORESTREGRESSION Train and evaluate Random Forest regression.
%
% This implementation uses MATLAB TreeBagger with OOB prediction.
%
% PIPELINE:
%   1. Data preprocessing (ensure column vector target)
%   2. Train Random Forest (TreeBagger)
%   3. Out-of-bag prediction (OOB)
%   4. Handle multivariate target via mean(Y,2)
%
% INPUTS:
%   X        - Feature matrix [nSamples x nFeatures]
%   y        - Target vector or matrix
%   numTrees - Number of decision trees
%
% OUTPUT:
%   results struct:
%       .model
%       .predictedValues
%       .originalValues
%       .oobModel

    %% Ensure correct shape
    if size(y,1) == 1
        y = y';
    end

    %% Reduce multivariate target
    if size(y,2) > 1
        yModel = mean(y,2);
    else
        yModel = y;
    end

    %% Train Random Forest
    rfModel = TreeBagger( ...
        numTrees, ...
        X, ...
        yModel, ...
        Method = 'regression', ...
        OOBPrediction = 'on' ...
    );

    %% Out-of-bag predictions
    yPred = oobPredict(rfModel);

    %% Output structure
    results = struct();

    results.model = rfModel;

    results.predictedValues = yPred;
    results.originalValues = yModel;

    results.oobModel = true;

end