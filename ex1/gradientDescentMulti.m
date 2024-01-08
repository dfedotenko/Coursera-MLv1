function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    for k = 1:size(X,2)

    sum = 0;
    for i = 1:m
        sum1 = 0;
        for j = 1:size(X,2)
          sum1 = sum1 + theta(j,1)*X(i,j);
        end
        sum = sum + ((sum1 - y(i,1)) * X(i,k));
    end    
    
    theta(k,1) = theta(k,1) - (alpha * (1/m * sum));

    end

    disp(computeCostMulti(X,y,theta));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
