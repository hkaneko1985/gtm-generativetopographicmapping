% Demonstration of GTM
clear; close all;

% settings
shapeofmap = [ 10, 10 ];
shapeofrbfcenters = [ 5, 5 ];
varianceofrbfs = 4;
lambdainemalgorithm = 0.001;
numberofiterations = 300;
displayflag = 1;

% load an iris dataset
load fisheriris
inputdataset = meas;

% autoscaling
inputdataset = zscore(inputdataset,1);

% construct GTM model
model = calc_gtm(inputdataset, shapeofmap,shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, displayflag);

if model.successflag
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

    % plot the mode of the responsibility
    [~, maxindex] = max(responsibilities, [], 2);
    modes = model.mapgrids(maxindex, :);
    figure; gscatter( modes(:,1), modes(:,2), species);
    axis square;
    xlabel( 'z_1 (mode)' ,  'FontSize' , 18 , 'FontName','Meiryo UI');
    ylabel( 'z_2 (mode)' ,  'FontSize' , 18 , 'FontName','Meiryo UI');
    set(gcf, 'Color' , 'w' ); 
    set(gca, 'FontSize' , 18);
    set(gca, 'FontName','Meiryo UI');
    axis([-1.1 1.1 -1.1 1.1]);
end
