function ypred = crossvalidationprediction(regressionfunction, X, y, parameters, foldnumber, rescalingflag)
% cross-validation
%    Hiromasa Kaneko
%
% --- input ---
% regressionfunction : function name of a used regression method
% X : m x n matrix of X-variables of training dataset (m is the number of samples and n is the number of X-variables)
% y : m x 1 vector of a y-variable of training dataset
% parameters : hyper-parameters of a used regressiom method
% foldnumber : fold number in cross-validation
% rescalingflag : with or without rescaling ( 1 or 0 )
% 
% --- output ---
% ypred : m x 1 vector of y-values estimated in cross-validation

rng('default')
fold = cvpartition( length(y), 'KFold', foldnumber);
ypred = zeros( length(y), 1);
for partition = 1 : foldnumber
    Xtrain = X(training(fold,partition), :);
    ytrain = y(training(fold,partition));
    Xtest = X(test(fold,partition), :);
    
    if rescalingflag
        % autoscaling
        autoscaledXtrain = ( Xtrain - repmat(mean(Xtrain),size(Xtrain,1),1) ) ./ repmat(std(Xtrain),size(Xtrain,1),1);
        autoscaledytrain = ( ytrain - mean(ytrain) ) ./ std(ytrain);
        autoscaledXtest = ( Xtest - repmat(mean(Xtrain),size(Xtest,1),1) ) ./ repmat(std(Xtrain),size(Xtest,1),1);
        ymean = mean(ytrain);
        ystd = std(ytrain);
    else
        autoscaledXtrain = Xtrain;
        autoscaledytrain = ytrain;
        autoscaledXtest = Xtest;
        ymean = 0;
        ystd = 1;
    end
    
    ypredtmp = feval(regressionfunction, autoscaledXtrain, autoscaledytrain, autoscaledXtest, parameters);
    ypred( test(fold, partition) ) = ypredtmp * ystd + ymean;
end
rng('shuffle');

end

