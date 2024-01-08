function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% Loop

%sum2 = 0;

% for i = 1:m
    %sum1 = theta(1,1);
    %for k = 2:size(X,2)
    %  sum1 = sum1 + theta(k,1) * X(i,k);
    %end
%    sum2 = sum2 + ( ( (-y(i)) * log(sigmoid( sum(sum(theta'.*X(i,:))) )) ) - ((1 - y(i)) * log(1 - sigmoid( sum(sum(theta'.*X(i,:))) ))) );
%end

%J = (1/m) * sum2;

% Vectorization
k = 1:m;
F = log(sigmoid(sum((theta'.*X(k,:)),2)));
G = log(1 - sigmoid(sum((theta'.*X(k,:)),2)));

% cost
J = (1/m) * sum( ( (-y).* F ) - ((1 - y).* G) );


% Loop
%for j = 1:size(X,2)

%sum2 = 0;

%for i = 1:m
    %sum1 = theta(1,1);
    %for k = 2:size(X,2)
    %  sum1 = sum1 + theta(k,1) * X(i,k);
    %end
%    sum2 = sum2 + ((sigmoid( sum(sum(theta'.*X(i,:))) ) - y(i)) * X(i,j));
%end

% gradient
% grad(j,1) = 1/m * sum2(1,1);

%end

% Vectorization: I could rewrite it into a single line but did not for
k = 1:m;
N = sigmoid(sum((theta'.*X(k,:)),2));
grad(:,1) = (1/m) * sum( ((N(k)- y(k)).* X(k,:)) );
% =============================================================


end