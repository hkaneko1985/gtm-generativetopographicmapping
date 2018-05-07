function likelihood = calc_likelihood(model, inputdataset)
%CALC_RESPONSIBILITY calculate likelihood

% distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W) .^ 2;
distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias) .^ 2;
% distancebetweenphiWandinputdataset = pdist2( inputdataset, [model.phiofmaprbfgrids ones(prod(model.shapeofmap), 1)]*model.W) .^ 2;
likelihood = sum( log( (model.beta/2/pi)^(size(inputdataset,2)/2) / prod(model.shapeofmap) * sum(exp(-model.beta/2*(distancebetweenphiWandinputdataset)),2)));
