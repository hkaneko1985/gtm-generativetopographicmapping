function [ypred, model] = gtmr_calc( X, y, Xtest, parameters)
% Modeling of Generative Topographic Mapping Regression (GTMR)
%   Hiromasa Kaneko
%
% --- input ---
% X : m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
% y : m x 1 vector of a y-variable of training dataset
% Xtest : k x n matrix of X-variables of test dataset
% parameters : parameters of GTM [shape of map, shape of RBF centers, variance of RBFs, lambda in EM algorithm, displayflag]
% 
% --- output ---
% ypred : m x 1 vector of predicted y-values
% model : constructed GTMR model

% settings
shapeofmap = [ parameters(1), parameters(1) ];
shapeofrbfcenters = [ parameters(2), parameters(2) ];
varianceofrbfs = parameters(3);
lambdainemalgorithm = parameters(4);
numberofiterations = parameters(5);
displayflag = parameters(6);
if length(parameters) > 6
    numberofy = parameters(7);
else
    numberofy = 1;
end

% construct GTM model
model = calc_gtm([X y], shapeofmap,shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, displayflag);

% prediction
ypred = gtmr_predict(model, Xtest);

end

