# -*- coding: utf-8 -*- 
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of GTMR (Generative Topographic Mapping Regression)

# settings
shapeofmap = [30, 30]
shapeofrbfcenters = [4, 4]
varianceofrbfs = 0.5
lambdainemalgorithm = 0.001
numberofiterations = 300
desplayflag = 1
numofsamples = 1000
noisey = 0.1
random_state_number = 30000

import numpy as np
#import pandas as pd
from sklearn.datasets.samples_generator import make_swiss_roll
from gtm import gtm
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from mpl_toolkits.mplot3d import Axes3D

# load a swiss roll dataset and make a y-variable
OriginalX, color = make_swiss_roll(numofsamples, 0, random_state=10)
X = OriginalX
rawy = 0.3 * OriginalX[:,0] - 0.1 * OriginalX[:,1] + 0.2 * OriginalX[:,2]
originaly = rawy + noisey * rawy.std(ddof=1) * np.random.randn(len(rawy))
# plot
plt.rcParams["font.size"] = 18
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(OriginalX[:,0], OriginalX[:,1], OriginalX[:,2], c=originaly)
fig.colorbar(p)
plt.show()

# divide a dataset into training data and test data
Xtrain = OriginalX[:500,:]
ytrain = originaly[:500]
Xtest = OriginalX[500:,:]
ytest = originaly[500:]

# autoscaling
#autoscaledX = (OriginalX - OriginalX.mean(axis=0)) / OriginalX.std(axis=0,ddof=1)
autoscaledXtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0,ddof=1)
autoscaledXtest = (Xtest - X.mean(axis=0)) / X.std(axis=0,ddof=1)
autoscaledytrain = (ytrain - ytrain.mean()) / ytrain.std(ddof=1)

# construct GTMR model
inputdataset = np.c_[ autoscaledXtrain, autoscaledytrain]
model = gtm( shapeofmap, shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, desplayflag)
model.fit(inputdataset)
if model.successflag:
    # calculate of responsibilities
    responsibilities = model.responsibility(inputdataset)
    
    # plot the mean of responsibilities
    means = responsibilities.dot( model.mapgrids )
    plt.figure()
#    plt.figure(figsize=figure.figaspect(1))
    plt.scatter( means[:,0], means[:,1], c=ytrain)
    plt.colorbar()
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1,1.1)
    plt.xlabel("z1 (mean)")
    plt.ylabel("z2 (mean)")
    plt.show()
    
    # plot the mode of responsibilities
    modes = model.mapgrids[responsibilities.argmax(axis=1), :]
    plt.figure()
#    plt.figure(figsize=figure.figaspect(1))
    plt.scatter( modes[:,0], modes[:,1], c=ytrain)
    plt.colorbar()
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1,1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()

# GTMR prediction
predictedytest_mean, predictedytest_mode, responsibilities, px = model.gtmrpredict( autoscaledXtest)
predictedytest = predictedytest_mean * ytrain.std(ddof=1) + ytrain.mean() # mean
#predictedytest = predictedytest_mode * ytrain.std(ddof=1) + ytrain.mean() # mode
predictedytest = np.ndarray.flatten(predictedytest)

# r2p, RMSEp, MAEp
print( "r2p: {0}".format(float( 1 - sum( (ytest-predictedytest )**2 ) / sum((ytest-ytest.mean())**2) )) )
print( "RMSEp: {0}".format(float( ( sum( (ytest-predictedytest)**2 )/ len(ytest))**(1/2) )) )
print( "MAEp: {0}".format(float( sum( abs(ytest-predictedytest)) / len(ytest) )) )
# yyplot
plt.figure(figsize=figure.figaspect(1))
plt.scatter( ytest, predictedytest)
YMax = np.max( np.array([np.array(ytest), predictedytest]))
YMin = np.min( np.array([np.array(ytest), predictedytest]))
plt.plot([YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], [YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin)], 'k-')
plt.ylim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlim(YMin-0.05*(YMax-YMin),YMax+0.05*(YMax-YMin))
plt.xlabel("simulated y")
plt.ylabel("estimated y")
plt.show()
