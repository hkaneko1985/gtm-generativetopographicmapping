function likelihood = calc_sgtm_likelihood(model, inputdataset)
%CALC_RESPONSIBILITY calculate likelihood

% distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W) .^ 2;
distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias) .^ 2;
% distancebetweenphiWandinputdataset = pdist2( inputdataset, [model.phiofmaprbfgrids ones(prod(model.shapeofmap), 1)]*model.W) .^ 2;
likelihood = sum( log( ((model.beta/2/pi)^(size(inputdataset,2)/2)) * sum(exp(-model.beta/2*(distancebetweenphiWandinputdataset)).* repmat( model.mixcoef, size(distancebetweenphiWandinputdataset,1), 1), 2)) );


% likelihood = sum( log( ((model.beta/2/pi)^(size(inputdataset,2)/2)) * sum(exp(-model.beta/2*(distancebetweenphiWandinputdataset)).* repmat( model.mixcoef, size(distancebetweenphiWandinputdataset,1), 1))) * 10^10 );
% likelihood = sum( log( (sum(exp(-model.beta/2*(distancebetweenphiWandinputdataset)).* repmat( model.mixcoef, size(distancebetweenphiWandinputdataset,1), 1)))));

% responsibilities = exp(-model.beta/2*(distancebetweenphiWandinputdataset)) .* repmat( model.mixcoef, size(distancebetweenphiWandinputdataset,1), 1);
% likelihood = sum(sum(responsibilities));