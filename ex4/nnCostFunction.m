function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Identity matrix of 10 labels
c = eye(num_labels);
y1 = zeros(m, num_labels);
i = 1:m; y1(i,:) = c(y(i),:);

% compute cost

% Loop
%{
sum2 = 0;
sum1 = 0;

for i = 1:m

    A = zeros(1,hidden_layer_size);
    B = zeros(1,num_labels);
    for a = 1:hidden_layer_size

     sum1 = 0;
     for k = 1:size(X,2)
       sum1 = sum1 + Theta1(a,k) * X(i,k);
     end
     A(1,a) = sigmoid(sum1);

    end

    A = [1, A];

    for b = 1:num_labels

     sum1 = 0;
     for k = 1:size(A,2)
       sum1 = sum1 + Theta2(b,k) * A(1,k);
     end
     B(1,b) = sigmoid(sum1);

    end

    sum1 = 0;
    for k = 1:num_labels
        sum1 = sum1 + (( ( (-c(y(i),k)) * log(B(1,k)) ) - ((1 - c(y(i),k)) * log(1-B(1,k))) ));
    end 
    sum2 = sum2 + sum1;

end
J = (1/m) * sum2;
%}

% Improved loop with vectorization
%{
sum2 = 0;
sum1 = 0;
for i = 1:m

    % vectorized - re-use predict.m from Exe3 for feed-forward
    F = ( (sigmoid(Theta2 * ([ ones(1, 1) (sigmoid(Theta1*X(i,:)'))' ])' )) );
    
    sum2 = 0;
    for k = 1:num_labels
        % Index into Identity matrix of 10 labels is the label from y at
        % index i of feature X(i,:)
        sum2 = sum2 + (( ( (-c(y(i),k)) * log(F(k,1)) ) - ((1 - c(y(i),k)) * log(1-F(k,1))) ));
    end  
    sum1 = sum1 + sum2;
end
J = (1/m) * sum1;
%}

% Complete vectorization
F = (sigmoid(Theta2 * ([ ones(m, 1) (sigmoid(Theta1*X'))' ])' ));
J = ( (1/m) * sum(sum(( ( (-y1').*log(F) ) - ((1 - y1').* log(1-F)) ))) );

% regularization - vectorized

% drop the 1st column of the bias
i = 2:size(Theta1,2);
sumReg1 = sum(sum(Theta1(:,i).^2));
i = 2:size(Theta2,2);
sumReg2 = sum(sum(Theta2(:,i).^2));

% Regularized cost
J = J + ((lambda / (2*m)) * (sumReg1 + sumReg2));

% Backpropagation
% Improved loop with vectorization

for i = 1:m

    % vectorized - re-use predict.m from Exe3 for feed-forward
    % F = ( (sigmoid(Theta2 * ([ ones(1, 1) (sigmoid(Theta1*X(i,:)'))' ])' )) );

    L2z = Theta1*X(i,:)'; % hidden layer
    L2a = ([ ones(1, 1) (sigmoid(L2z))' ]); % hidden layer
    L3z = Theta2 * L2a'; % output layer
    L3a = ( sigmoid( L3z ) ); % output layer
   
    %L3 - out layer gradient calc: 
    % Loop
    %{
    for k = 1:num_labels
        % Index into Identity matrix of 10 labels is the label from y at
        % index i of feature X(i,:)
        sigma2(k,:) = (L3a(k,:) - c(y(i),1));
    end
    %}

    % Vectorized
    sigma2 = L3a - y1(i,:)';
    sigma1 = ((Theta2(:,2:end))'*sigma2).*sigmoidGradient(L2z);

    Theta2_grad = Theta2_grad + (sigma2 * L2a);
    Theta1_grad = Theta1_grad + (sigma1 * (X(i,:)));
end

% Final gradients
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularized gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * (Theta1(:,2:end)));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * (Theta2(:,2:end)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
