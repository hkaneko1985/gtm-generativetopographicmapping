# -*- coding: utf-8 -*- 
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# GTM (generative topographic mapping) class
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA

class gtm:
    def __init__( self, shapeofmap=[30,30], shapeofrbfcenters=[10,10], varianceofrbfs=4, lambdainemalgorithm=0.001, numberofiterations=200, displayflag=1):
        self.shapeofmap = shapeofmap
        self.shapeofrbfcenters = shapeofrbfcenters
        self.varianceofrbfs = varianceofrbfs
        self.lambdainemalgorithm = lambdainemalgorithm
        self.numberofiterations = numberofiterations
        self.displayflag = displayflag
    
    def fit( self, inputdataset):
        inputdataset = np.array(inputdataset)
        self.successflag = True
        # make rbf grids
        [rbfgridsx, rbfgridsy] = np.meshgrid( np.arange( 0.0, self.shapeofrbfcenters[0]), np.arange( 0.0, self.shapeofrbfcenters[1]) );
        self.rbfgrids = np.c_[np.ndarray.flatten(rbfgridsx)[:,np.newaxis], np.ndarray.flatten(rbfgridsy)[:,np.newaxis] ]
        maxrbfgrids= self.rbfgrids.max(axis=0)        
        self.rbfgrids[:,0] = 2*( self.rbfgrids[:,0] - maxrbfgrids[0]/2 ) / maxrbfgrids[0]
        self.rbfgrids[:,1] = 2*( self.rbfgrids[:,1] - maxrbfgrids[1]/2 ) / maxrbfgrids[1]
        
        # make map grids
        [mapgridsx, mapgridsy] = np.meshgrid( np.arange( 0.0, self.shapeofmap[0]), np.arange( 0.0, self.shapeofmap[1]) );
        self.mapgrids = np.c_[np.ndarray.flatten(mapgridsx)[:,np.newaxis], np.ndarray.flatten(mapgridsy)[:,np.newaxis] ]
        maxmapgrids= self.mapgrids.max(axis=0)        
        self.mapgrids[:,0] = 2*( self.mapgrids[:,0] - maxmapgrids[0]/2 ) / maxmapgrids[0]
        self.mapgrids[:,1] = 2*( self.mapgrids[:,1] - maxmapgrids[1]/2 ) / maxmapgrids[1]
        
        # calculate phi of mapgrids and rbfgrids
        distancebetweenmapandrbfgrids = distance.cdist( self.mapgrids, self.rbfgrids, 'sqeuclidean' )
        self.phiofmaprbfgrids = np.exp( -distancebetweenmapandrbfgrids / 2.0 / self.varianceofrbfs )
        
        # PCA for initializing W and beta
        pcamodel = PCA( n_components = 3 )
        pcamodel.fit_transform(inputdataset)
        if np.linalg.matrix_rank(self.phiofmaprbfgrids) < min(self.phiofmaprbfgrids.shape):
            self.successflag = False
            return
        self.W = np.linalg.pinv(self.phiofmaprbfgrids).dot( self.mapgrids.dot( pcamodel.components_[0:2,:] ) )
        self.beta = min( pcamodel.explained_variance_[2], 1/( (distance.cdist(self.phiofmaprbfgrids.dot(self.W), self.phiofmaprbfgrids.dot(self.W)) + np.diag(np.ones(np.prod(self.shapeofmap))*10**100)).min(axis=0).mean()/2 ) )
        self.bias = inputdataset.mean(axis=0)
        
        # EM algorithm
        phiofmaprbfgridswithone = np.c_[self.phiofmaprbfgrids, np.ones((np.prod(self.shapeofmap), 1))]
        for iteration in range(self.numberofiterations):
            responsibilities = self.responsibility(inputdataset)
            if responsibilities.sum() == 0:
                self.successflag = False
                break
                
            phitGphietc = phiofmaprbfgridswithone.T.dot( np.diag(responsibilities.sum(axis=0)).dot(phiofmaprbfgridswithone)) + self.lambdainemalgorithm/self.beta * np.identity(phiofmaprbfgridswithone.shape[1])
            if 1/np.linalg.cond(phitGphietc) < 10**-15:
                self.successflag = False
                break
            self.Wwithone = np.linalg.inv(phitGphietc).dot( phiofmaprbfgridswithone.T.dot(responsibilities.T.dot(inputdataset)))
            self.beta = inputdataset.size / ( responsibilities * distance.cdist( inputdataset, phiofmaprbfgridswithone.dot(self.Wwithone))**2 ).sum()

            self.W = self.Wwithone[:-1,:]
            self.bias = self.Wwithone[-1,:]

            if self.displayflag:
                print( "{0}/{1} ... likelihood: {2}".format( iteration+1, self.numberofiterations, self.likelihood( inputdataset)) )
            
    def responsibility( self, inputdataset):
        inputdataset = np.array(inputdataset)
        distancebetweenphiWandinputdataset = distance.cdist( inputdataset, self.phiofmaprbfgrids.dot(self.W) + np.ones((np.prod(self.shapeofmap),1)).dot(np.reshape(self.bias,(1,len(self.bias)))), 'sqeuclidean')
        rbfforresponsibility = np.exp(-self.beta/2.0*(distancebetweenphiWandinputdataset))
        sumrbfforresponsibility = rbfforresponsibility.sum(axis=1)
#        return rbfforresponsibility / np.reshape( sumrbfforresponsibility, (rbfforresponsibility.shape[0],1))
        if np.count_nonzero(sumrbfforresponsibility) == len(sumrbfforresponsibility):
            return rbfforresponsibility / np.reshape( sumrbfforresponsibility, (rbfforresponsibility.shape[0],1))
        else:
            return np.zeros(rbfforresponsibility.shape)
            
    def likelihood( self, inputdataset):
        inputdataset = np.array(inputdataset)
        distancebetweenphiWandinputdataset = distance.cdist( inputdataset, self.phiofmaprbfgrids.dot(self.W) + np.ones((np.prod(self.shapeofmap),1)).dot(np.reshape(self.bias,(1,len(self.bias)))), 'sqeuclidean')
        return ( np.log( (self.beta/2.0/np.pi)**(inputdataset.shape[1]/2.0) / np.prod(self.shapeofmap) * ((np.exp(-self.beta/2.0*(distancebetweenphiWandinputdataset))).sum(axis=0)) )).sum()
