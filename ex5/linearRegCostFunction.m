function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% from ex3 modified for linear vs logistic hypothesis f, by only removing
% sigmoid f in favour of square error sum
% Vectorization
k = 1:m;
F = sum((theta'.*X(k,:)),2);

k = 2:size(X,2);
sumReg = sum(theta(k,1).^2);

% cost
J = ((1/(2*m)) * sum( (F - y).^2 )) + ((lambda / (2*m)) * sumReg);

% =============================================================

% from ex3 modified for linear vs logistic hypothesis f, by only removing
% sigmoid f in favour of square error sum
% Vectorization
k = 1:m;
N = sum((theta'.*X(k,:)),2);
% gradient

grad(:,1) = ((1/m) * sum( ((N(k)- y(k)).* X(k,:)) ));

l = 2:size(X,2);
grad(l,1) = grad(l,1) + (((lambda / m) * theta(l,1)));

% =========================================================================

grad = grad(:);

end
