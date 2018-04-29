function [xpred_mean, xpred_mode, wk, pyzs] = inverse_gtmmlr(targetyvalues, gtmmodel, regressioncoefficients, sigma)
% Inverse analysis of GTM-MLR (Generative Topographic Mapping - Multiple Linear Regression)
%    Hiromasa Kaneko
%
% --- input ---
% targetyvalues : m x 1 vector of target y-values (n is the number of target y-values)
% gtmmodel : GTM model constructed using "gtmmlr.m"
% regressioncoefficients, sigma : MLR model constructed using "gtmmlr.m"
% 
% --- output ---
% xpred_mean : m x n matrix of mean of predicted X-variables (n is the number of X-variables)
% xpred_mode : m x n matrix of mode of predicted X-variables (n is the number of X-variables)
% wk : responsibilities, which can be used to discussed assigned grids on the GTM map
% pyzs : m x k matrix of probability of y given zi ( k is the map size), which can be used to discuss applicability domains

myu_i = gtmmodel.phiofmaprbfgrids*gtmmodel.W + ones(prod(gtmmodel.shapeofmap), 1)*gtmmodel.bias;
sigmai = zeros( length(regressioncoefficients) );
sigmai(1:size(sigmai,1)+1:end) = 1/gtmmodel.beta;

invsigmai = zeros( length(regressioncoefficients) );
invsigmai(1:size(invsigmai,1)+1:end) = gtmmodel.beta;

deltai = inv( invsigmai + regressioncoefficients*regressioncoefficients'/sigma );

xpred_mean = []; xpred_mode = []; 
% maxes = [];
for targetyvalue = targetyvalues
    pxy_means = zeros( size(myu_i) );
    for i = 1 : size(pxy_means,1)
        pxy_means(i,:) = deltai * ( regressioncoefficients/sigma*targetyvalue + invsigmai*myu_i(i,:)');
    end

    pyz_means = myu_i*regressioncoefficients;
    pyz_var = sigma + regressioncoefficients'*sigmai*regressioncoefficients;
    pyzs = zeros( length(pyz_means), 1);
    for i = 1 : length(pyzs)
        pyzs(i) = normpdf( targetyvalue, pyz_means(i), sqrt(pyz_var));
    end
    wk = pyzs/ sum(pyzs);

%     randomcentersofx = rand( samplingnumber, size(X,2) );
%     randomcentersofx = randomcentersofx .* repmat(max(X)-min(X), size(randomcentersofx,1), 1 );
%     randomcentersofx = randomcentersofx + repmat(min(X), size(randomcentersofx,1), 1 );
%     proby = zeros( samplingnumber, 1 );
%     for i = 1 : samplingnumber
%         for j = 1 : length(wk)
%             proby(i) = proby(i) + wk(j)*mvnpdf( randomcentersofx(i,:), pxy_means(j,:), deltai);
%         end
%     end

    meanofx = wk' * pxy_means;
    modeofx = pxy_means(wk==max(wk),:);

%     maxofx = randomcentersofx(proby==max(proby),:);
    
    xpred_mean = [ xpred_mean; meanofx];
    xpred_mode = [ xpred_mode; modeofx];
%     maxes = [ maxes; maxofx ];
    
end

end

