# -*- coding: utf-8 -*- 
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of inverse GTM-MLR (Generative Topographic Mapping - Multiple Linear Regression)

targetyvalue = 4 # y-target for inverse analysis

# settings
shapeofmap = [30, 30]
shapeofrbfcenters = [4, 4]
varianceofrbfs = 0.5
lambdainemalgorithm = 0.001
numberofiterations = 200
desplayflag = 1
k = 10
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

# autoscaling
autoscaledX = (OriginalX - OriginalX.mean(axis=0)) / OriginalX.std(axis=0,ddof=1)

# construct GTM model
model = gtm( shapeofmap, shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, desplayflag)
model.fit(autoscaledX)
if model.successflag:
    # calculate of responsibilities
    responsibilities = model.responsibility(autoscaledX)
    
    # plot the mean of responsibilities
    means = responsibilities.dot( model.mapgrids )
    plt.figure()
#    plt.figure(figsize=figure.figaspect(1))
    plt.scatter( means[:,0], means[:,1], c=originaly)
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
    plt.scatter( modes[:,0], modes[:,1], c=originaly)
    plt.colorbar()
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1,1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()

# construct MLR model
model.mlr(OriginalX,originaly)

# inverse analysis
estimatedxmean, estimatedxmode, responsibilities_inverse = model.inversegtmmlr( targetyvalue)
estimatedxmean = estimatedxmean * OriginalX.std(axis=0,ddof=1) + OriginalX.mean(axis=0)
estimatedxmode = estimatedxmode * OriginalX.std(axis=0,ddof=1) + OriginalX.mean(axis=0)
print("estimated x-mean: {0}".format(estimatedxmean))
print("estimated x-mode: {0}".format(estimatedxmode))

estimatedxmean_onmap = responsibilities_inverse.dot( model.mapgrids )
estimatedxmode_onmap = model.mapgrids[np.argmax(responsibilities_inverse),:]
print("estimated x-mean on map: {0}".format(estimatedxmean_onmap))
print("estimated x-mode on map: {0}".format(estimatedxmode_onmap))
