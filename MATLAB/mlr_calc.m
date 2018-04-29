function [ypred, regressioncoefficients] = mlr_calc(Xtrain, ytrain, Xtest, ~)
%MLR_CALC multiple linear regression or ordinary least squares
%   Xtrain and ytrain must be autoscaled.

regressioncoefficients = Xtrain'*Xtrain \ Xtrain' * ytrain;
ypred = Xtest * regressioncoefficients;

end

