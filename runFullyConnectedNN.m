function results = runFullyConnectedNN( ...
    X, ...
    y, ...
    fc1Units, ...
    fc2Units, ...
    fc3Units, ...
    maxEpochs, ...
    miniBatchSize, ...
    gradientThreshold ...
)
%RUNFULLYCONNECTEDNN Train and evaluate fully connected feedforward NN
%
% INPUTS:
%   X                 - [nSamples x nFeatures]
%   y                 - target vector/matrix
%   fc*Units          - hidden layer sizes
%   maxEpochs         - training epochs
%   miniBatchSize     - batch size
%   gradientThreshold - gradient clipping threshold
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

    %% Reduce multivariate target if needed
    if size(yTrain,2) > 1
        yTrainModel = mean(yTrain,2);
        yTestModel  = mean(yTest,2);
    else
        yTrainModel = yTrain;
        yTestModel  = yTest;
    end

    %% Network architecture (pure MLP)
    numFeatures = size(X,2);
    numResponses = 1;

    layers = [
        featureInputLayer(numFeatures)

        fullyConnectedLayer(fc1Units)
        reluLayer

        fullyConnectedLayer(fc2Units)
        reluLayer

        fullyConnectedLayer(fc3Units)
        reluLayer

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
        'Verbose',0);

    %% Train model
    net = trainNetwork(XTrain, yTrainModel, layers, options);

    %% Predict
    yPred = predict(net, XTest, 'MiniBatchSize', 1);

    %% Output struct
    results = struct();
    results.model = net;

    results.predictedValues = yPred;
    results.originalValues = yTestModel;

    results.trainData = struct('X', XTrain, 'y', yTrainModel);
    results.testData  = struct('X', XTest,  'y', yTestModel);

end