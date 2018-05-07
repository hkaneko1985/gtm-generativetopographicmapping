clear; close all;
% Demonstration of inverse GTMR (Generative Topographic Mapping Regression) using a swiss roll dataset

targetyvalue = 4; % y-target for inverse analysis

% settings
shapeofmap = [ 30, 30];
shapeofrbfcenters = [ 4, 4];
varianceofrbfs = 2;
lambdainemalgorithm = 0.0001;
numberofiterations = 300;
displayflag = 1;
k = 10;

% load a dataset
dataset = csvread('swissroll.csv');
y = dataset(:,3);
X = dataset(:,4:end);

figure;
scatter3(X(:,1),X(:,2),X(:,3),[],y,'filled');
colormap(jet);
colorbar;
xlim([ min(X(:,1))-range(X(:,1))*0.03 max(X(:,1))+range(X(:,1))*0.03] );
ylim([ min(X(:,2))-range(X(:,2))*0.03 max(X(:,2))+range(X(:,2))*0.03] );
zlim([ min(X(:,3))-range(X(:,3))*0.03 max(X(:,3))+range(X(:,3))*0.03] );
axis square;
xlabel( 'x_1' ,  'FontSize' , 18 , 'FontName', 'Times');
ylabel( 'x_2' ,  'FontSize' , 18 , 'FontName', 'Times');
zlabel( 'x_3' ,  'FontSize' , 18 , 'FontName', 'Times');
set(gcf, 'Color' , 'w' ); 
set(gca, 'FontSize', 18);
set(gca, 'FontName', 'Times');

% autoscaling
autoscaledX = ( X - repmat(mean(X),size(X,1),1) ) ./ repmat(std(X),size(X,1),1);
autoscaledy = ( y - mean(y) ) ./ std(y);

% construct GTM model
[~, gtmmodel] = gtmr_calc( autoscaledX, autoscaledy, autoscaledX, [shapeofmap(1) shapeofrbfcenters(1) varianceofrbfs  lambdainemalgorithm numberofiterations 0]);
if gtmmodel.successflag
    % calculate responsibility
    responsibilities = calc_responsibility(gtmmodel, [autoscaledX autoscaledy]);
    % plot the mean of the responsibility
    means = responsibilities * gtmmodel.mapgrids;
    figure; scatter( means(:,1), means(:,2), [], y, 'filled');
    colormap(jet); colorbar;
    axis square; 
    xlabel( 'z_1 (mean)' ,  'FontSize' , 18 , 'FontName','Times');
    ylabel( 'z_2 (mean)' ,  'FontSize' , 18 , 'FontName','Times');
    set(gcf, 'Color' , 'w' ); 
    set(gca, 'FontSize' , 18);
    set(gca, 'FontName','Times');
    axis([-1.1 1.1 -1.1 1.1]);

    % plot the mode of the responsibility
    [~, maxindex] = max(responsibilities, [], 2);
    modes = gtmmodel.mapgrids(maxindex, :);
    figure; scatter( modes(:,1), modes(:,2), [], y, 'filled');
    colormap(jet); colorbar;
    axis square;
    xlabel( 'z_1 (mode)' ,  'FontSize' , 18 , 'FontName','Times');
    ylabel( 'z_2 (mode)' ,  'FontSize' , 18 , 'FontName','Times');
    set(gcf, 'Color' , 'w' ); 
    set(gca, 'FontSize' , 18);
    set(gca, 'FontName','Times');
    axis([-1.1 1.1 -1.1 1.1]);
end

% Inverse analysis
[means_autoscaled, modes_autoscaled, responsibilities_inverse] = inverse_gtmr( gtmmodel, ( targetyvalue - mean(y) ) ./ std(y));
xpred_mean = means_autoscaled .* std(X) + mean(X)
xpred_mode = modes_autoscaled .* std(X) + mean(X) 

xpred_mean_on_map = responsibilities_inverse * gtmmodel.mapgrids
[~, maxindex] = max(responsibilities_inverse);
xpred_mode_on_map = gtmmodel.mapgrids(maxindex, :)
