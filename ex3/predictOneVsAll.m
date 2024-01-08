function [p] = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

G = zeros(size(all_theta, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% run over all examples in X
% for each example X(i,:), run over all_theta's
% for each theta, compute a sigmoid and record the max sigmoid
% for each p(i) record the max(sigmoid) >= 0.5 = 1 otherwise, = 0;

% Loop
% for all examples in X
%for k = 1:m
    
    % for each example X(i,:), run over all_theta's  
%    for m = 1:num_labels

%     sum1 = 0;
%     for j = 1:size(all_theta, 2)
%       sum1 = sum1 + all_theta(m,j) * X(k,j);
%     end

%     G(m) = sigmoid(sum1);

%    end

    % for each p(i) record the max(sigmoid) position; 
%    [M, I] = max(G, [], 1);
%    p(k) = I;   

  
%end

% Vectorized
[M, I] = max(sigmoid(all_theta*X'));
p = I';

% =========================================================================

end
