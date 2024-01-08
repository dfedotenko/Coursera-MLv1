function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Iterate over centroids
for k = 1:K
    count = find(idx == k);
    % Vectorized
    j = 1:size(count,1); centroids(k,:) = sum(X(count(j),:))/size(count,1);

    % Loop
    %new_c = zeros(1,n);
    % Iterate over centroid indeces
    %for j = 1:size(count,1)
        % for each centroid count the # of occurences and sum it up
    %    new_c = new_c + X(count(j),:);  
    %end   
    % compute new centroid using sum over count
    % centroids(k,:) = new_c./size(count,1);
end    

% =============================================================


end

