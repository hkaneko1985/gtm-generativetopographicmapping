function [gtmmodel, regressioncoefficients, sigma, calculatedy, estimatedycv] = gtmmlr(X, y, foldnumber, candidatesofshapeofmap, candidatesofshapeofrbfcenters, candidatesofvarianceofrbfs, candidatesoflambdainemalgorithm, numberofiterations, k, displayflag, rescalingflag)
% Modeling of GTM-MLR (Generative Topographic Mapping - Multiple Linear Regression)
%    Hiromasa Kaneko
%
% --- input ---
% X : m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
% y : m x 1 vector of a y-variable of training dataset
% foldnumber : fold number in cross-validation
% candidatesofshapeofmap : candidates of shape of map
% candidatesofshapeofrbfcenters : candidates of shape of RBF centers
% candidatesofvarianceofrbfs : candidates of variance of RBFs, lambda in EM algorithm, displayflag]
% candidatesoflambdainemalgorithm : candidates of lambda in EM algorithm
% numberofiterations : number of iterations in EM algorithm
% k : k in k3n-error
% displayflag: with or without display ( 1 or 0 )
% rescalingflag : with or without rescaling ( 1 or 0 )
% 
% --- output ---
% gtmmodel : constructed GTM model
% regressioncoefficients, sigma : constructed MLR model
% calculatedy : m x 1 vector of y-values calculated
% estimatedycv : m x 1 vector of y-values estimated in cross-validation

%% grid search for GTM
% candidates of GTM hyperparameters
gridcell = cell(1,4);
gridcell{1}=candidatesofshapeofmap; gridcell{2}=candidatesofshapeofrbfcenters; gridcell{3}=candidatesofvarianceofrbfs; gridcell{4}=candidatesoflambdainemalgorithm;
gridparameters = gridcell{1}'; 
for pnum = 2 : size( gridcell, 2 )
    grid_seed = gridcell{pnum};
    gridtmp = repmat( grid_seed, size( gridparameters, 1 ), 1 );
    gridparameters = [ repmat( gridparameters, size(grid_seed,2), 1 ) gridtmp(:) ];        
end

k3nerrorofGTM = zeros(size(gridparameters,1),1); %k3nerror
for gridnumber = 1 : size(gridparameters,1)
    % construct GTM model
    gtmmodel = calc_gtm(X, [gridparameters(gridnumber,1), gridparameters(gridnumber,1)], [gridparameters(gridnumber,2), gridparameters(gridnumber,2)], gridparameters(gridnumber,3), gridparameters(gridnumber,4), numberofiterations, 0);
    if gtmmodel.successflag
        % calculate responsibility
        responsibilities = calc_responsibility(gtmmodel, X);
        % calculate the mean of the responsibility
        means = responsibilities * gtmmodel.mapgrids;
        k3nerrorofGTM(gridnumber) = k3nerror( X, means, k) + knnnormalizeddist( means, X, k);
    else
        k3nerrorofGTM(gridnumber) = 10^100;
    end
    if displayflag
        disp( [gridnumber size(gridparameters,1)])
    end
end

%% optimized GTM
optimizedhyperparametermnumber = find( k3nerrorofGTM == min(k3nerrorofGTM) );
shapeofmap = [gridparameters(optimizedhyperparametermnumber,1), gridparameters(optimizedhyperparametermnumber,1)];
shapeofrbfcenters = [gridparameters(optimizedhyperparametermnumber,2), gridparameters(optimizedhyperparametermnumber,2)];
varianceofrbfs = gridparameters(optimizedhyperparametermnumber,3);
lambdainemalgorithm = gridparameters(optimizedhyperparametermnumber,4);

%% construct GTM model
gtmmodel = calc_gtm(X, shapeofmap,shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, 0);

%% construct MLR model
[calculatedy, regressioncoefficients] = mlr_calc(X, y, X);
estimatedycv = crossvalidationprediction(@mlr_calc, X, y, [], foldnumber, rescalingflag);
sigma = 1/length(y) * sum( (y - calculatedy).^2 );

end

