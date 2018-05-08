# -*- coding: utf-8 -*-
#%reset -f
"""
@author: Hiromasa Kaneko
"""

# GTM (generative topographic mapping) class
import numpy as np
from scipy.spatial import distance
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA


class gtm:

    def __init__(self, shapeofmap=[30,30], shapeofrbfcenters=[10,10],
                 varianceofrbfs=4, lambdainemalgorithm=0.001,
                 numberofiterations=200, displayflag=1):
        self.shapeofmap = shapeofmap
        self.shapeofrbfcenters = shapeofrbfcenters
        self.varianceofrbfs = varianceofrbfs
        self.lambdainemalgorithm = lambdainemalgorithm
        self.numberofiterations = numberofiterations
        self.displayflag = displayflag

    def calculate_grids(self, num_x, num_y):
        grids_x, grids_y = np.meshgrid(np.arange(0.0, num_x), np.arange(0.0, num_y))
        grids = np.c_[np.ndarray.flatten(grids_x)[:, np.newaxis],
                      np.ndarray.flatten(grids_y)[:, np.newaxis]]
        max_grids = grids.max(axis=0)
        grids[:, 0] = 2 * (grids[:, 0]-max_grids[0]/2) / max_grids[0]
        grids[:, 1] = 2 * (grids[:, 1]-max_grids[1]/2) / max_grids[1]
        return grids

    def fit(self, inputdataset):
        # inputdataset: numpy.array or pandas.DataFrame
        # inputdataset must be autoscaled.
        inputdataset = np.array(inputdataset)
        self.successflag = True
        # make rbf grids
        self.rbfgrids = self.calculate_grids(self.shapeofrbfcenters[0],
                                             self.shapeofrbfcenters[1])

        # make map grids
        self.mapgrids = self.calculate_grids(self.shapeofmap[0],
                                             self.shapeofmap[1])

        # calculate phi of mapgrids and rbfgrids
        distancebetweenmapandrbfgrids = distance.cdist(self.mapgrids,
                                             self.rbfgrids, 'sqeuclidean')
        self.phiofmaprbfgrids = np.exp(-distancebetweenmapandrbfgrids / 2.0
                                        / self.varianceofrbfs)

        # PCA for initializing W and beta
        pcamodel = PCA(n_components=3)
        pcamodel.fit_transform(inputdataset)
        if np.linalg.matrix_rank(self.phiofmaprbfgrids) < min(self.phiofmaprbfgrids.shape):
            self.successflag = False
            return
        self.W = np.linalg.pinv(self.phiofmaprbfgrids).dot(
                         self.mapgrids.dot(pcamodel.components_[0:2, :]))
        self.beta = min(pcamodel.explained_variance_[2], 1/(
                        (
                            distance.cdist(self.phiofmaprbfgrids.dot(self.W),
                            self.phiofmaprbfgrids.dot(self.W))
                            + np.diag(np.ones(np.prod(self.shapeofmap))*10**100)
                        ).min(axis=0).mean()/2))
        self.bias = inputdataset.mean(axis=0)

        # EM algorithm
        phiofmaprbfgridswithone = np.c_[self.phiofmaprbfgrids,
                                        np.ones((np.prod(self.shapeofmap), 1))]
        for iteration in range(self.numberofiterations):
            responsibilities = self.responsibility(inputdataset)
            if responsibilities.sum() == 0:
                self.successflag = False
                break

            phitGphietc = phiofmaprbfgridswithone.T.dot(
                 np.diag(responsibilities.sum(axis=0)).dot(phiofmaprbfgridswithone)
                 ) + self.lambdainemalgorithm/self.beta * np.identity(
                phiofmaprbfgridswithone.shape[1])
            if 1/np.linalg.cond(phitGphietc) < 10**-15:
                self.successflag = False
                break
            self.Wwithone = np.linalg.inv(phitGphietc).dot(
                 phiofmaprbfgridswithone.T.dot(responsibilities.T.dot(inputdataset)))
            self.beta = inputdataset.size / (responsibilities
                  * distance.cdist(inputdataset,
                   phiofmaprbfgridswithone.dot(self.Wwithone))**2).sum()

            self.W = self.Wwithone[:-1, :]
            self.bias = self.Wwithone[-1, :]

            if self.displayflag:
                print("{0}/{1} ... likelihood: {2}".format(
                     iteration+1, self.numberofiterations,
                     self.likelihood(inputdataset)))

    def responsibility(self, inputdataset):
        # inputdataset: numpy.array or pandas.DataFrame
        # inputdataset must be autoscaled.
        inputdataset = np.array(inputdataset)
        distancebetweenphiWandinputdataset = distance.cdist(
           inputdataset, self.phiofmaprbfgrids.dot(self.W) 
           + np.ones((np.prod(self.shapeofmap), 1)).dot(
           np.reshape(self.bias, (1, len(self.bias)))), 'sqeuclidean')
        rbfforresponsibility = np.exp(-self.beta/2.0
                                   *(distancebetweenphiWandinputdataset))
        sumrbfforresponsibility = rbfforresponsibility.sum(axis=1)
#        return rbfforresponsibility / np.reshape( sumrbfforresponsibility, (rbfforresponsibility.shape[0],1))
        if np.count_nonzero(sumrbfforresponsibility) == len(sumrbfforresponsibility):
            return rbfforresponsibility / np.reshape(sumrbfforresponsibility,
                        (rbfforresponsibility.shape[0], 1))
        else:
            return np.zeros(rbfforresponsibility.shape)

    def likelihood(self, inputdataset):
        # inputdataset must be autoscaled.
        inputdataset = np.array(inputdataset)
        distancebetweenphiWandinputdataset = distance.cdist(
               inputdataset, self.phiofmaprbfgrids.dot(self.W) +
               np.ones((np.prod(self.shapeofmap),1)).dot(
               np.reshape(self.bias, (1, len(self.bias)))), 'sqeuclidean')
        return (np.log((self.beta/2.0/np.pi)**(inputdataset.shape[1]/2.0) /
                 np.prod(self.shapeofmap) * ((
                 np.exp(-self.beta/2.0*(distancebetweenphiWandinputdataset))
                ).sum(axis=1)) )).sum()

    def mlr(self, X, y):
        # X, y: numpy.array or pandas.DataFrame
        # Both X and y must NOT be autoscaled.
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))
        # autoscaling
        self.Xmean = X.mean(axis=0)
        self.Xstd = X.std(axis=0, ddof=1)
        autoscaledX = (X - self.Xmean) / self.Xstd
        self.ymean = y.mean(axis=0)
        self.ystd = y.std(axis=0, ddof=1)
        autoscaledy = (y - self.ymean) / self.ystd
        self.regressioncoefficients = np.linalg.inv(
          np.dot(autoscaledX.T, autoscaledX)).dot(autoscaledX.T.dot(autoscaledy))
        calculatedy = autoscaledX.dot(self.regressioncoefficients) * self.ystd + self.ymean
        self.sigma = sum((y - calculatedy)**2) / len(y)

    def mlrpredict(self, X):
        # X: numpy.array or pandas.DataFrame
        # X must NOT be autoscaled.
        autoscaledX = (X -  self.Xmean) / self.Xstd
        return autoscaledX.dot(self.regressioncoefficients) * self.ystd + self.ymean

    def inversegtmmlr(self, targetyvalue):
        # targetvalue must be scaler.
#        targetyvalues = np.ndarray.flatten(np.array(targetyvalues))
        myu_i = self.phiofmaprbfgrids.dot(self.W) + np.ones(
        (np.prod(self.shapeofmap), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))
        sigmai = np.diag(np.ones(len(self.regressioncoefficients))) / self.beta
        invsigmai = np.diag(np.ones(len(self.regressioncoefficients))) * self.beta
        deltai = np.linalg.inv(invsigmai 
          + self.regressioncoefficients.dot(self.regressioncoefficients.T) / self.sigma)
#        for targetyvalue in targetyvalues:
        pxy_means = np.empty(myu_i.shape)
        for i in range(pxy_means.shape[0]):
            pxy_means[i,:] = np.ndarray.flatten(
                 deltai.dot( 
                    self.regressioncoefficients/self.sigma*targetyvalue
                    + invsigmai.dot(np.reshape(myu_i[i,:], [myu_i.shape[1], 1]))
                 ))

        pyz_means = myu_i.dot(self.regressioncoefficients)
        pyz_var = self.sigma + self.regressioncoefficients.T.dot(
                                 sigmai.dot(self.regressioncoefficients))
        pyzs = np.empty(len(pyz_means))
        for i in range(len(pyz_means)):
            pyzs[i] = norm.pdf(targetyvalue, pyz_means[i], pyz_var**(1/2))

        responsibilities_inverse = pyzs / pyzs.sum()
        estimatedxmean = responsibilities_inverse.dot(pxy_means)
        estimatedxmode = pxy_means[ np.argmax(responsibilities_inverse), : ]

        # responsibilities_inverse can be used to discussed assigned grids on the GTM map
        # pyzs : vector of probability of y given zi, which can be used to discuss applicability domains
        return estimatedxmean, estimatedxmode, responsibilities_inverse

    def gtmrpredict(self, X):
        # X: numpy.array or pandas.DataFrame
        # X must be autoscaled.
        # Multiple y-variables are OK.
        # In model, the rigth p variables are handled as y-variables ( p is the number of y-variables ).

        if self.successflag:
            X = np.array(X)
            myu_i = self.phiofmaprbfgrids.dot(self.W) + np.ones(
                                          (np.prod(self.shapeofmap), 1)
                            ).dot(np.reshape(self.bias, (1, len(self.bias))))
            delta_x = np.diag(np.ones(X.shape[1])) / self.beta
            px = np.empty([X.shape[0], myu_i.shape[0]])
            for i in range(myu_i.shape[0]):
                px[:,i] = multivariate_normal.pdf(X, myu_i[i, 0:X.shape[1]], delta_x)

            responsibilities = px.T / px.T.sum(axis=0)
            responsibilities = responsibilities.T
            estimatedymean = responsibilities.dot(myu_i[:, X.shape[1]:] )
            estimatedymode = myu_i[np.argmax(responsibilities, axis=1), X.shape[1]:]
        else:
            estimatedymean = np.zeros(X.spape[0])
            estimatedymode = np.zeros(X.spape[0])
            responsibilities = np.empty([X.shape[0], myu_i.shape[0]])

        # responsibilities can be used to discussed assigned grids on the GTM map
        # px [p(x)] : vector of probability of x given myu_x_i and sigma_x_i, which can be used to discuss applicability domains
        return estimatedymean, estimatedymode, responsibilities, px

    def inversegtmr(self, targetyvalue):
        # targetvalue must be one candidate.
        # But, multiple y-variables are OK.
        # In model, the rigth m variables are handled as X-variables ( m is the number of X-variables ).

        myu_i = self.phiofmaprbfgrids.dot(self.W) + np.ones(
           (np.prod(self.shapeofmap), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))
        delta_y = 1 / self.beta
        py = np.empty(myu_i.shape[0])
        if isinstance(targetyvalue,int) or isinstance(targetyvalue,float):
            for i in range(myu_i.shape[0]):
                py[i] = multivariate_normal.pdf(targetyvalue, myu_i[i, -1], delta_y)

            responsibilities_inverse = py / py.sum()
            estimatedxmean = responsibilities_inverse.dot( myu_i[:, 0:-1] )
            estimatedxmode = myu_i[np.argmax(responsibilities_inverse), 0:-1]
        else:
            for i in range(myu_i.shape[0]):
                py[i] = multivariate_normal.pdf(
                        targetyvalue, myu_i[i, -len(targetyvalue)], delta_y)

            responsibilities_inverse = py / py.sum()
            estimatedxmean = responsibilities_inverse.dot(myu_i[:, 0:-len(targetyvalue)])
            estimatedxmode = myu_i[np.argmax(responsibilities_inverse), 0:-len(targetyvalue)]

        # responsibilities_inverse can be used to discussed assigned grids on the GTM map
        # py [p(y)] : vector of probability of y given myu_y_i and sigma_y_i, which can be used to discuss applicability domains
        return estimatedxmean, estimatedxmode, responsibilities_inverse, py

