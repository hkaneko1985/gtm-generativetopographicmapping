function responsibilities = calc_sgtm_responsibility(model, inputdataset)
%CALC_RESPONSIBILITY calculate responsibility for input dataset

% distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W) .^ 2;
distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias) .^ 2;
% responsibilities = ((model.beta/2/pi)^(size(inputdataset,2)/2)) * exp(-model.beta/2*(distancebetweenphiWandinputdataset));
responsibilities = exp(-model.beta/2*(distancebetweenphiWandinputdataset)) .* repmat( model.mixcoef, size(distancebetweenphiWandinputdataset,1), 1);
sumresponsibilities = sum( responsibilities, 2 );
if any(sumresponsibilities) == 0
   responsibilities = [];
else
   responsibilities = responsibilities ./ repmat(sum( responsibilities, 2 ), 1, size(responsibilities,2) );
end
