function [xpred_mean, xpred_mode, w_ky, py] = inverse_gtmr( model, targetyvalues)
% Inverse analysis of Generative Topographic Mapping Regression (GTMR) model
% Multiple y-variables are OK.
% In model, the rigth p variables are handled as y-variables ( p is the number of y-variables ).
%   Hiromasa Kaneko
%
% --- input ---
% model : GTMR model constructed using "gtmr_calc.m"
% targetyvalues : m x p vector of target y-values (n is the number of target y-values and p is the number of y-variables.)
% 
% --- output ---
% xpred_mean : m x n matrix of mean of predicted X-variables (n is the number of X-variables)
% xpred_mode : m x n matrix of mode of predicted X-variables (n is the number of X-variables)
% w_ky : responsibilities, which can be used to discussed assigned grids on the GTM map
% py [p(y)] : m x k matrix of probability of y given myu_y_i and sigma_y_i ( k is the map size), which can be used to discuss applicability domains

myu_i = model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias;
delta_y = 1/model.beta;

myu_ky = myu_i(:,end-size(targetyvalues,2)+1:end);
myu_kx = myu_i(:,1:end-size(targetyvalues,2));

m_ky = myu_kx;

py = zeros( length(targetyvalues), size( myu_ky, 1 ) );
for j = 1 : length(py)
    py(:,j) = mvnpdf( targetyvalues, myu_ky(j), delta_y);
end
w_ky = py ./ repmat(sum(py,2), 1, length(targetyvalues));
xpred_mean = w_ky*m_ky;
[~,maxindex] = max(w_ky,[],2);
xpred_mode = m_ky(maxindex,:);

end

