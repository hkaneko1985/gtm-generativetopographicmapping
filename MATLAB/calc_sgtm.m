function model = calc_sgtm(inputdataset, shapeofmap,...
    shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, displayflag)
%GTM_CALC calculation of GTM

model.shapeofmap = shapeofmap;
model.shapeofrbfcenters = shapeofrbfcenters;
model.varianceofrbfs = varianceofrbfs;
model.lambdainemalgorithm = lambdainemalgorithm;
model.numberofiterations = numberofiterations;
model.successflag = 1;

%make rbf grids
[rbfgridsx, rbfgridsy] = meshgrid((0:1:(shapeofrbfcenters(1)-1)), ((shapeofrbfcenters(2)-1):-1:0));
model.rbfgrids = [rbfgridsx(:), rbfgridsy(:)];
maxrbfgrids= max(model.rbfgrids);
model.rbfgrids(:,1) = 2*(model.rbfgrids(:,1) - maxrbfgrids(1)/2)./maxrbfgrids(1);
model.rbfgrids(:,2) = 2*(model.rbfgrids(:,2) - maxrbfgrids(2)/2)./maxrbfgrids(2);

%make map grids
[mapgridsx, mapgridsy] = meshgrid((0:1:(shapeofmap(1)-1)), ((shapeofmap(2)-1):-1:0));
model.mapgrids = [mapgridsx(:), mapgridsy(:)];
maxmapgrids= max(model.mapgrids);
model.mapgrids(:,1) = 2*(model.mapgrids(:,1) - maxmapgrids(1)/2)./maxmapgrids(1);
model.mapgrids(:,2) = 2*(model.mapgrids(:,2) - maxmapgrids(2)/2)./maxmapgrids(2);

%calculate phi of mapgrids and rbfgrids
distancebetweenmapandrbfgrids = pdist2(model.mapgrids, model.rbfgrids) .^ 2;
model.phiofmaprbfgrids = exp( - distancebetweenmapandrbfgrids /2/varianceofrbfs );

%PCA for initializing W and beta
[ ~, S , eigenvectors ] = svd( inputdataset );
eigenvalues = diag( S .^ 2 ) / (size(inputdataset, 1)-1);
if rank(model.phiofmaprbfgrids) < min(size(model.phiofmaprbfgrids))
    model.successflag = 0;
    return;
end
model.W = model.phiofmaprbfgrids \ (model.mapgrids*eigenvectors(:,1:2)');
% model.W = model.phiofmaprbfgrids \ (zscore(model.mapgrids)*eigenvectors(:,1:2)');
model.beta = min( 1/eigenvalues(3), 1/(mean(min(pdist2(model.phiofmaprbfgrids*model.W, model.phiofmaprbfgrids*model.W) + diag(ones(size(model.phiofmaprbfgrids*model.W,1), 1)*10^100)))/2) );
model.bias = mean( inputdataset );
model.mixcoef = ones( 1, prod(model.shapeofmap)) / prod(model.shapeofmap); % mixing coefficient

%EM algorithm
phiofmaprbfgridswithone = [model.phiofmaprbfgrids ones(prod(model.shapeofmap), 1)];
sumofabsW_old = 0;
beta_old = 0;
for iteration = 1 : numberofiterations
    responsibilities = calc_sgtm_responsibility(model, inputdataset);
    if isempty(responsibilities)
        model.successflag = 0;
        break;
    end
    
    phitGphietc = (phiofmaprbfgridswithone'*diag(sum(responsibilities))*phiofmaprbfgridswithone + lambdainemalgorithm/model.beta*diag(ones(size(phiofmaprbfgridswithone,2) , 1)));
%     if rcond(phitGphietc) < max(size(phitGphietc))*eps(norm(phitGphietc))
%         model.successflag = 0;
%         break;
%     end
    
    model.Wwithone = phitGphietc \ ...
        (phiofmaprbfgridswithone'*responsibilities'*inputdataset);
    model.beta = numel(inputdataset) / sum(sum( responsibilities .* pdist2( inputdataset, phiofmaprbfgridswithone*model.Wwithone).^2 ));

    model.W = model.Wwithone(1:end-1,:);
    model.bias = model.Wwithone(end,:);
    model.mixcoef = sum(responsibilities)/size(inputdataset,1);
    
    likelihood = calc_likelihood(model, inputdataset);
    if displayflag
        disp( [ num2str(iteration) ' / ' num2str( numberofiterations ) ' ... likelihood: ' num2str(likelihood) ] );
    end
    
%     sumofabsW = abs(sum(sum(model.W)));
    sumofabsW = abs(sum(sum(model.Wwithone)));
    if abs( sumofabsW - sumofabsW_old ) + abs( model.beta - beta_old ) < 10^-10
        break;
    end
    sumofabsW_old = sumofabsW;
    beta_old = model.beta;
end

end

