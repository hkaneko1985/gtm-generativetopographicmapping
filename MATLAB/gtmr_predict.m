function [ypred_mean, ypred_mode, w_kx, px] = gtmr_predict(model, X )
% Predict new X using GTMR model
% Multiple y-variables are OK.
% In model, the left m variables are handled as X-variables ( m is the
% number of X-variables ).
%   Hiromasa Kaneko
%
% --- input ---
% model : GTMR model constructed using "gtmr_calc.m"
% X : m x n matrix of target y-values (m is the number of samples and n is the number of X-variables)
% 
% --- output ---
% ypred_mean : m x numberofy vector of mean of predicted X-variables (n is the number of X-variables)
% ypred_mode : m x numberofy vector of mode of predicted X-variables (n is the number of X-variables)
% w_kx : responsibilities, which can be used to discussed assigned grids on the GTM map
% px [p(x)] : m x k matrix of probability of x given myu_x_i and sigma_x_i ( k is the map size), which can be used to discuss applicability domains

if model.successflag
    myu_i = model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias;
    delta_x = zeros( size(X,2) );
    delta_x(1:size(delta_x,1)+1:end) = 1/model.beta;
    % deltay = 1/beta;
    % pii = 1/size(myui,1);

    myu_ky = myu_i(:,size(X,2)+1:end);
    myu_kx = myu_i(:,1:size(X,2));

    m_kx = myu_ky;
    px = zeros( size(X,1), size(myu_kx,1) );
    for i = 1 : size(myu_kx,1)
        px(:,i) = mvnpdf( X, myu_kx(i,:), delta_x);
    end
    w_kx = px ./ repmat(sum(px,2), 1, size(px,2));
    ypred_mean = w_kx*m_kx;
    [~,maxindex] = max(w_kx,[],2);
    ypred_mode = m_kx(maxindex);
else
    ypred_mode = zeros( size(X,1), 1);
    ypred_mean = zeros( size(X,1), 1);
    w_kx = zeros( size(X,1), prod(model.shapeofmap));
    px = zeros( size(X,1), prod(model.shapeofmap));
end

end

