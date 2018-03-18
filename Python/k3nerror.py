# k3n-error
import numpy as np

# k-nearest neighbor normalized error (k3n-error)
# When X1 is data of X-variables and X2 is data of Z-variables (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
# When X1 is Z-variables (low-dimensional data) and X2 is data of data of X-variables, this is k3n error in reconstruction (k3n-X-error).
# k3n-error = k3n-Z-error + k3n-X-error

# X1, X2: numpy.array or pandas.DataFrame
# k: integer, the numbers of neighbor

def k3nerror(X1, X2, k):
    sumofk3nerror = 0
    X1 = np.array(X1)
    X2 = np.array(X2)
    for samplenumber in range(X1.shape[0]):
        X1distance = np.ndarray.flatten( np.sqrt( ( ( X1[:, np.newaxis] - X1[samplenumber,:])**2).sum(axis=2)))
        X1sortedindex = np.delete( np.argsort(X1distance), 0)
        
        X2distance = np.ndarray.flatten( np.sqrt( ( ( X2[:, np.newaxis] - X2[samplenumber,:])**2).sum(axis=2)))
        X2sortedindex = np.delete( np.argsort(X2distance), 0)
        X2distance[ X2distance==0 ] = np.min( X2distance[ X2distance !=0])
        
        sumofk3nerror = sumofk3nerror + ((np.sort(X2distance[X1sortedindex[0:k]]) - X2distance[ X2sortedindex[0:k]] ) / X2distance[X2sortedindex[0:k]]).sum()
    
    return sumofk3nerror / X1.shape[0] / k