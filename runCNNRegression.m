function results = runCNNRegression( ...
    X, ...
    y, ...
    conv1Filters, ...
    conv2Filters, ...
    conv3Filters, ...
    fc1Units, ...
    fc2Units, ...
    maxEpochs, ...
    miniBatchSize, ...
    gradientThreshold ...
)
%RUNCNNREGRESSION Train and evaluate 1D CNN regression model
%
% INPUTS:
%   X               - [nSamples x nFeatures]
%   y               - target vector/matrix
%   conv*Filters    - convolution layer sizes
%   fc*Units        - fully connected layer sizes
%   maxEpochs       - training epochs
%   miniBatchSize   - batch size
%   gradientThreshold - gradient clipping
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

    %% CNN architecture
    numFeatures = size(X,2);
    numResponses = 1;

    layers = [
        sequenceInputLayer(numFeatures)

        convolution1dLayer(conv1Filters, numFeatures, "Padding","causal")
        reluLayer

        convolution1dLayer(conv2Filters, numFeatures*2, "Padding","same")
        reluLayer

        convolution1dLayer(conv3Filters, numFeatures, "Padding","same")
        reluLayer

        fullyConnectedLayer(fc1Units)
        fullyConnectedLayer(fc2Units)

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

    %% Convert to sequence format
    XTrainSeq = XTrain';
    XTestSeq  = XTest';

    %% Train model
    net = trainNetwork(XTrainSeq, yTrainModel', layers, options);

    %% Predict
    yPred = predict(net, XTestSeq, 'MiniBatchSize', 1);
    yPred = yPred';

    %% Output
    results = struct();
    results.model = net;

    results.predictedValues = yPred;
    results.originalValues = yTestModel;

    results.trainData = struct('X', XTrain, 'y', yTrainModel);
    results.testData  = struct('X', XTest,  'y', yTestModel);

end