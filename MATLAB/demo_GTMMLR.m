clear; close all;
% Demonstration of GTM-MLR (Generative Topographic Mapping - Multiple Linear Regression) using a swiss roll dataset

targetyvalue = 4; % y-target for inverse analysis

% settings
shapeofmap = [ 30, 30];
shapeofrbfcenters = [ 4, 4];
varianceofrbfs = 0.5;
lambdainemalgorithm = 0.001;
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

% divide a dataset into training data and test data
traindatanumber = 500;
ytrain = y(1:traindatanumber,:); Xtrain = X(1:traindatanumber,:);
ytest = y(traindatanumber+1:end,:); Xtest = X(traindatanumber+1:end,:);

% autoscaling
autoscaledXtrain = ( Xtrain - repmat(mean(Xtrain),size(Xtrain,1),1) ) ./ repmat(std(Xtrain),size(Xtrain,1),1);
autoscaledytrain = ( ytrain - mean(ytrain) ) ./ std(ytrain);
autoscaledXtest = ( Xtest - repmat(mean(Xtrain),size(Xtest,1),1) ) ./ repmat(std(Xtrain),size(Xtest,1),1);

% construct GTM model
gtmmodel = calc_gtm(autoscaledXtrain, shapeofmap,shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, 0);
if gtmmodel.successflag
    % calculate responsibility
    responsibilities = calc_responsibility(gtmmodel, autoscaledXtrain);
    % plot the mean of the responsibility
    xpred_mean = responsibilities * gtmmodel.mapgrids;
    figure; scatter( xpred_mean(:,1), xpred_mean(:,2), [], ytrain, 'filled');
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
    xpred_mode = gtmmodel.mapgrids(maxindex, :);
    figure; scatter( xpred_mode(:,1), xpred_mode(:,2), [], ytrain, 'filled');
    colormap(jet); colorbar;
    axis square;
    xlabel( 'z_1 (mode)' ,  'FontSize' , 18 , 'FontName','Times');
    ylabel( 'z_2 (mode)' ,  'FontSize' , 18 , 'FontName','Times');
    set(gcf, 'Color' , 'w' ); 
    set(gca, 'FontSize' , 18);
    set(gca, 'FontName','Times');
    axis([-1.1 1.1 -1.1 1.1]);
end
% construct MLR model
[calculatedy, regressioncoefficients] = mlr_calc(autoscaledXtrain, autoscaledytrain, autoscaledXtrain);
calculatedy = calculatedy * std(ytrain) + mean(ytrain);
sigma = 1/length(y) * sum( (ytrain - calculatedy).^2 );
% MLR prediction
ytestpred = autoscaledXtest * regressioncoefficients;
ytestpred = ytestpred * std(ytrain) + mean(ytrain);
figure;
plot( ytest, ytestpred, 'b.', 'MarkerSize' , 15 );
hold on;
plot( [-6.5 5.5] , [-6.5 5.5] , 'k' , 'LineWidth' , 2 );
xlabel('simulated y' , 'FontSize' , 18 , 'FontName', 'Times');
ylabel('estimated y' , 'FontSize' , 18 , 'FontName', 'Times');
axis( [ -6.5 5.5 -6.5 5.5 ] );
hold off;
set(gcf, 'Color' , 'w' );
set(gca, 'FontSize' ,20 );
axis square;
r2p = 1 - ( ( ytest - ytestpred )' * ( ytest - ytestpred ) ) ./ ( ( ytest - mean(ytest) )' * ( ytest - mean(ytest) ) );
maep = sum( abs( ytest - ytestpred ) ) / length(ytest);
disp( 'r2p and MAEp' );
disp( [ r2p, maep] );
