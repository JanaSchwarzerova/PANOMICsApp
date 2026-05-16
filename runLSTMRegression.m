function results = runLSTMRegression( ...
    X, ...
    y, ...
    lstmUnits, ...
    fcUnits, ...
    dropoutRate, ...
    maxEpochs, ...
    miniBatchSize, ...
    gradientThreshold ...
)
%RUNLSTMREGRESSION Train and evaluate LSTM regression model.
%
% This implementation uses MATLAB deep learning toolbox.
%
% PIPELINE:
%   1. Train/test split (80/20 chronological)
%   2. LSTM network definition
%   3. Training with trainingOptions
%   4. Prediction on test set
%
% NOTE:
%   - Multivariate Y is reduced using mean(Y,2)
%   - Input is treated as sequence data (features -> time steps)
%
% INPUTS:
%   X                 - Feature matrix [nSamples x nFeatures]
%   y                 - Target vector or matrix
%   lstmUnits        - Number of LSTM units
%   fcUnits          - Fully connected layer size
%   dropoutRate      - Dropout probability
%   maxEpochs        - Training epochs
%   miniBatchSize    - Batch size
%   gradientThreshold - Gradient clipping threshold
%
% OUTPUT:
%   results struct:
%       .model
%       .predictedValues
%       .originalValues
%       .trainData
%       .testData

    %% Ensure correct shape
    if size(y,1) == 1
        y = y';
    end

    %% Train/test split (80/20 chronological)
    n = size(X,1);

    splitIdx = floor(0.8 * n);

    XTrain = X(1:splitIdx,:);
    XTest  = X(splitIdx+1:end,:);

    yTrain = y(1:splitIdx,:);
    yTest  = y(splitIdx+1:end,:);

    %% Reduce multivariate target
    if size(yTrain,2) > 1
        yTrainModel = mean(yTrain,2);
        yTestModel  = mean(yTest,2);
    else
        yTrainModel = yTrain;
        yTestModel  = yTest;
    end

    %% Network architecture
    numFeatures = size(X,2);
    numResponses = 1;

    layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(lstmUnits, 'OutputMode','sequence')
        fullyConnectedLayer(fcUnits)
        dropoutLayer(dropoutRate)
        fullyConnectedLayer(numResponses)
        regressionLayer
    ];

    %% Training options
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','cpu', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'GradientThreshold', gradientThreshold, ...
        'Shuffle','never', ...
        'Verbose',0 ...
    );

    %% Convert to sequence format (features as time steps)
    XTrainSeq = XTrain';
    XTestSeq  = XTest';

    %% Train model
    net = trainNetwork(XTrainSeq, yTrainModel', layers, options);

    %% Predict
    yPred = predict(net, XTestSeq, 'MiniBatchSize', 1);
    yPred = yPred';

    %% Output struct
    results = struct();

    results.model = net;

    results.predictedValues = yPred;
    results.originalValues = yTestModel;

    results.trainData = struct('X', XTrain, 'y', yTrainModel);
    results.testData  = struct('X', XTest,  'y', yTestModel);

end