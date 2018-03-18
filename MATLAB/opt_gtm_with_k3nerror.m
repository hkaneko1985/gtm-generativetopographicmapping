% Demonstration of optimization of GTM hyperparameters with k3nerror
clear; close all;

% settings
candidatesofshapeofmap = 30;
candidatesofshapeofrbfcenters = 2:2:20;
candidatesofvarianceofrbfs = 2.^(-5:2:3);
candidatesoflambdainemalgorithm = [0 10.^(-4:-1)];
numberofiterations = 200;
displayflag = 0;
k = 10;

% load an iris dataset
load fisheriris
inputdataset = meas;

% autoscaling
inputdataset = zscore(inputdataset,1);

%% grid search
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
    model = calc_gtm(inputdataset, [gridparameters(gridnumber,1), gridparameters(gridnumber,1)], [gridparameters(gridnumber,2), gridparameters(gridnumber,2)], gridparameters(gridnumber,3), gridparameters(gridnumber,4), numberofiterations, displayflag);
    if model.successflag
        % calculate responsibility
        responsibilities = calc_responsibility(model, inputdataset);
        % calculate the mean of the responsibility
        means = responsibilities * model.mapgrids;
        k3nerrorofGTM(gridnumber) = k3nerror( inputdataset, means, k) + knnnormalizeddist( means, inputdataset, k);
    else
        k3nerrorofGTM(gridnumber) = 10^100;
    end
    disp( [gridnumber size(gridparameters,1)])
end

%% optimized GTM
optimizedhyperparametermnumber = find( k3nerrorofGTM == min(k3nerrorofGTM) );
shapeofmap = [gridparameters(optimizedhyperparametermnumber,1), gridparameters(optimizedhyperparametermnumber,1)];
shapeofrbfcenters = [gridparameters(optimizedhyperparametermnumber,2), gridparameters(optimizedhyperparametermnumber,2)];
varianceofrbfs = gridparameters(optimizedhyperparametermnumber,3);
lambdainemalgorithm = gridparameters(optimizedhyperparametermnumber,4);

% construct GTM model
model = calc_gtm(inputdataset, shapeofmap,shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, displayflag);

% calculate responsibility
responsibilities = calc_responsibility(model, inputdataset);

% plot the mean of the responsibility
means = responsibilities * model.mapgrids;
figure; gscatter( means(:,1), means(:,2), species);
axis square; 
xlabel( 'z_1 (mean)' ,  'FontSize' , 18 , 'FontName','Meiryo UI');
ylabel( 'z_2 (mean)' ,  'FontSize' , 18 , 'FontName','Meiryo UI');
set(gcf, 'Color' , 'w' ); 
set(gca, 'FontSize' , 18);
set(gca, 'FontName','Meiryo UI');
axis([-1.1 1.1 -1.1 1.1]);

disp( 'Optimized hyperparameters' );
fprintf('Optimal mapsize: %d, %d\n', shapeofmap(1), shapeofmap(2));
fprintf('Optimal shape of RBF centers: %d, %d\n', shapeofrbfcenters(1), shapeofrbfcenters(2));
fprintf('Optimal variance of RBFs: %g\n', varianceofrbfs);
fprintf('Optimal lambda in EM algorithm: %g\n', lambdainemalgorithm);
