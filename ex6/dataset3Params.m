function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmav = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

minError = 1;

for i = 1:size(Cv,2)
    for j = 1:size(sigmav,2)
        % Train the model on the training set with a given choice of C and
        % sigma
        model = svmTrain(X, y, Cv(i), @(x1, x2)gaussianKernel(x1, x2, sigmav(j)));
        % Run predictions for a model in question on the cross-val set
        predictions = svmPredict(model, Xval);
        % Evaluation predictions accuracy against the actual set
        % store the prediction in the vector
        val = mean(double(predictions ~= yval));
        if val < minError
           C = Cv(i);
           sigma = sigmav(j);
           minError = val;
        end    
    end
end

% =========================================================================

end
