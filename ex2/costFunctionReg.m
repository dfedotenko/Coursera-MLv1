function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% cost function
% Loop
%sum = 0;
%for i = 1:m
    % first compute linear sum of theta and x
%    sum1 = theta(1,1);
%    for k = 2:size(X,2)
%      sum1 = sum1 + theta(k,1) * X(i,k);
%    end
    % theta' * X(i)
%    sum = sum + ( ( (-y(i)) * log(sigmoid(sum1)) ) - ((1 - y(i)) * log(1 - sigmoid(sum1))) );
%end

%sumReg = 0;
%for k = 2:size(X,2)
%    sumReg = sumReg + theta(k,1)^2;
%end

% cost
%J = (((1/m) * sum) + ((lambda / (2*m)) * sumReg));

% Vectorization
k = 1:m;
F = log(sigmoid(sum((theta'.*X(k,:)),2)));
G = log(1 - sigmoid(sum((theta'.*X(k,:)),2)));

k = 2:size(X,2);
sumReg = sum(theta(k,1).^2);

% cost
J = ((1/m) * sum( ( (-y).* F ) - ((1 - y).* G) )) + ((lambda / (2*m)) * sumReg);



% gradient calculation

%for j = 1:size(X,2)

%sum2 = 0;

%for i = 1:m
    % first compute linear sum of theta and x
%    sum1 = theta(1,1);
%    for k = 2:size(X,2)
%      sum1 = sum1 + theta(k,1) * X(i,k);
%    end
    % theta' * X(i)
%    sum2 = sum2 + ((sigmoid(sum1) - y(i)) * X(i,j));
%end

% gradient
%if j == 1
%   grad(j,1) = 1/m * sum2(1,1);    
%else
%   grad(j,1) = 1/m * sum2(1,1) + ((lambda / m) * theta(j,1));
%end    

%end

% Vectorization: I could rewrite it into a single line but did not for
% clarity
k = 1:m;
N = sigmoid(sum((theta'.*X(k,:)),2));
grad(:,1) = ((1/m) * sum( ((N(k)- y(k)).* X(k,:)) ));

l = 2:size(X,2);
grad(l,1) = grad(l,1) + (((lambda / m) * theta(l,1)));
% =============================================================

end
