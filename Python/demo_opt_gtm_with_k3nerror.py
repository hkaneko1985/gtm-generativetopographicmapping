# -*- coding: utf-8 -*- 
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of optimization of GTM hyperparameters with k3nerror

# settings
import numpy as np
candidatesofshapeofmap = np.arange( 30, 31, dtype=int)
candidatesofshapeofrbfcenters = np.arange( 2, 22, 2, dtype=int)
candidatesofvarianceofrbfs = 2**np.arange( -5, 4, 2, dtype=float)
candidatesoflambdainemalgorithm = 2**np.arange( -4, 0, dtype=float)
candidatesoflambdainemalgorithm = np.append( 0, candidatesoflambdainemalgorithm)
numberofiterations = 200
desplayflag = 0
k = 10

from sklearn.datasets import load_iris
from gtm import gtm
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from k3nerror import k3nerror

# load an iris dataset
iris = load_iris()
#inputdataset = pd.DataFrame(iris.data, columns=iris.feature_names)
inputdataset = iris.data
color = iris.target

# autoscaling
inputdataset = (inputdataset - inputdataset.mean(axis=0)) / inputdataset.std(axis=0,ddof=1)

# grid search
gridparametersandk3nerror = []
allcalcnumber = len(candidatesofshapeofmap)*len(candidatesofshapeofrbfcenters)*len(candidatesofvarianceofrbfs)*len(candidatesoflambdainemalgorithm)
calcnumber = 0
for shapeofmapgrid in candidatesofshapeofmap:
    for shapeofrbfcentersgrid in candidatesofshapeofrbfcenters:
        for varianceofrbfsgrid in candidatesofvarianceofrbfs:
            for lambdainemalgorithmgrid in candidatesoflambdainemalgorithm:
                calcnumber = calcnumber + 1
                print([calcnumber, allcalcnumber])
                # construct GTM model
                model = gtm( [shapeofmapgrid,shapeofmapgrid], [shapeofrbfcentersgrid,shapeofrbfcentersgrid], varianceofrbfsgrid, lambdainemalgorithmgrid, numberofiterations, desplayflag)
                model.fit(inputdataset)
                if model.successflag:
                    # calculate of responsibilities
                    responsibilities = model.responsibility(inputdataset)
                    # calculate the mean of responsibilities
                    means = responsibilities.dot( model.mapgrids )
                    # calculate k3nerror
                    k3nerrorofgtm = k3nerror(inputdataset, means, k)
                else:
                    k3nerrorofgtm = 10**100
                gridparametersandk3nerror.append( [shapeofmapgrid, shapeofrbfcentersgrid, varianceofrbfsgrid, lambdainemalgorithmgrid, k3nerrorofgtm])
                
# optimized GTM
gridparametersandk3nerror = np.array(gridparametersandk3nerror)
optimizedhyperparametermnumber = np.where( gridparametersandk3nerror[:,4] == np.min(gridparametersandk3nerror[:,4]) )[0][0]
shapeofmap = [gridparametersandk3nerror[optimizedhyperparametermnumber,0], gridparametersandk3nerror[optimizedhyperparametermnumber,0]]
shapeofrbfcenters = [gridparametersandk3nerror[optimizedhyperparametermnumber,1], gridparametersandk3nerror[optimizedhyperparametermnumber,1]]
varianceofrbfs = gridparametersandk3nerror[optimizedhyperparametermnumber, 2]
lambdainemalgorithm = gridparametersandk3nerror[optimizedhyperparametermnumber, 3]

# construct GTM model
model = gtm( shapeofmap, shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, desplayflag)
model.fit(inputdataset)

# calculate of responsibilities
responsibilities = model.responsibility(inputdataset)

# plot the mean of responsibilities
means = responsibilities.dot( model.mapgrids )
plt.figure(figsize=figure.figaspect(1))
plt.scatter( means[:,0], means[:,1], c=color)
plt.ylim(-1.1,1.1)
plt.xlim(-1.1,1.1)
plt.xlabel("z1 (mean)")
plt.ylabel("z2 (mean)")
plt.show()

print("Optimized hyperparameters" )
print("Optimal mapsize: {0}, {1}".format(shapeofmap[0],shapeofmap[1]))
print("Optimal shape of RBF centers: {0}, {1}".format(shapeofrbfcenters[0],shapeofrbfcenters[1]))
print("Optimal variance of RBFs: {0}".format(varianceofrbfs))
print("Optimal lambda in EM algorithm: {0}".format(lambdainemalgorithm))
