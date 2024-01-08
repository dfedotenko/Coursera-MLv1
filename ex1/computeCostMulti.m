function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

sum = 0;
for i = 1:m
    sum1 = theta(1,1);
    for j = 2:size(X,2)
      sum1 = sum1 + theta(j,1) * X(i,j);
    end
    sum = sum + (sum1 - y(i,1)) ^ 2;
end

J = (1/(2*m)) * sum;


% =========================================================================

end
