function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

vals = zeros(K, 1);

% Iterate over X
for j = 1:size(X,1)
    % Find the closest centroid based on Euclidian distance
    for k = 1:K
       % norm is: sqrt(sum((X(j,:) - centroids(k,:)) .^ 2)); 
       vals(k) = norm(X(j,:) - centroids(k,:));    
    end
    % Store the index of the closest centroid against a given X(j)
    [M,I] = min(vals);
    idx(j) = I;
end

% =============================================================

end

