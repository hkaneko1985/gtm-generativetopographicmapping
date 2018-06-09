function responsibilities = calc_responsibility(model, inputdataset)
%CALC_RESPONSIBILITY calculate responsibility for input dataset

% distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W) .^ 2;
distancebetweenphiWandinputdataset = pdist2( inputdataset, model.phiofmaprbfgrids*model.W + ones(prod(model.shapeofmap), 1)*model.bias) .^ 2;
% responsibilities = ((model.beta/2/pi)^(size(inputdataset,2)/2)) * exp(-model.beta/2*(distancebetweenphiWandinputdataset));
responsibilities = exp(-model.beta/2*(distancebetweenphiWandinputdataset));
sumresponsibilities = sum( responsibilities, 2 );
if any(sumresponsibilities)
	zero_sample = find(sumresponsibilities == 0);
    sumresponsibilities(zero_sample) = 1;
    responsibilities(zero_sample,:) = 1/size(responsibilities,2);
end
responsibilities = responsibilities ./ repmat(sum( responsibilities, 2 ), 1, size(responsibilities,2) );
