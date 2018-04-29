# -*- coding: utf-8 -*- 
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# Demonstration of GTM

# settings
shapeofmap = [10, 10]
shapeofrbfcenters = [5, 5]
varianceofrbfs = 4
lambdainemalgorithm = 0.001
numberofiterations = 200
#numberofiterations = 10
desplayflag = 1

from sklearn.datasets import load_iris
from gtm import gtm
import matplotlib.pyplot as plt
import matplotlib.figure as figure

# load an iris dataset
iris = load_iris()
#inputdataset = pd.DataFrame(iris.data, columns=iris.feature_names)
inputdataset = iris.data
color = iris.target

# autoscaling
inputdataset = (inputdataset - inputdataset.mean(axis=0)) / inputdataset.std(axis=0,ddof=1)

# construct GTM model
model = gtm( shapeofmap, shapeofrbfcenters, varianceofrbfs, lambdainemalgorithm, numberofiterations, desplayflag)
model.fit(inputdataset)

if model.successflag:
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
    
    # plot the mode of responsibilities
    modes = model.mapgrids[responsibilities.argmax(axis=1), :]
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter( modes[:,0], modes[:,1], c=color)
    plt.ylim(-1.1,1.1)
    plt.xlim(-1.1,1.1)
    plt.xlabel("z1 (mode)")
    plt.ylabel("z2 (mode)")
    plt.show()
