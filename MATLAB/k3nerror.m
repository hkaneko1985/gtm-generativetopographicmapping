function k3nerror = k3nerror( X1, X2, k)
% k-nearest neighbor normalized error for visualization and reconstruction (k3nerror) ? a measure for data visualization performance
%
% k-nearest neighbor normalized error (k3n-error)
% When X1 is data of X-variables and X2 is data of Z-variables (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
% When X1 is Z-variables (low-dimensional data) and X2 is data of data of X-variables, this is k3n error in reconstruction (k3n-X-error).
% k3n-error = k3n-Z-error + k3n-X-error
%
% --- input ---
% X1 : X-variables ( m x n, m is the number of samples and n is the number of variables) or Z-variables ( m x 2 )
% X2 : Z-variables or X-variables
%
% --- output ---
% k3nerror : k-nearest neighbor normalized error
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k3nerror = 0;
for samplenum = 1 : size(X1,1)
    X1dist = pdist2( X1, X1(samplenum,:) );
    [ ~, X1smallestdistind] = sort( X1dist );
    X1smallestdistind(1) = [];
    X2dist = pdist2( X2, X2(samplenum,:) ) / size(X2,2);
    [ ~, X2smallestdistind] = sort( X2dist );
    X2smallestdistind(1) = [];
    X2dist( X2dist==0 ) = min( X2dist(X2dist~=0) );
    k3nerror = k3nerror + sum( (sort( X2dist( X1smallestdistind(1:k) ) ) - X2dist( X2smallestdistind(1:k) )) ./ X2dist( X2smallestdistind(1:k) ) );
end

k3nerror = k3nerror / size(X1,1) / k;
