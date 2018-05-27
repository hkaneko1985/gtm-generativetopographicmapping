# coding: utf-8
import numpy as np
from scipy.spatial import distance


def k3nerror(X1, X2, k):
    """
    k-nearest neighbor normalized error (k3n-error)

    When X1 is data of X-variables and X2 is data of Z-variables
    (low-dimensional data), this is k3n error in visualization (k3n-Z-error).
    When X1 is Z-variables (low-dimensional data) and X2 is data of data of
    X-variables, this is k3n error in reconstruction (k3n-X-error).

    k3n-error = k3n-Z-error + k3n-X-error

    Parameters
    ----------
    X1: numpy.array or pandas.DataFrame
    X2: numpy.array or pandas.DataFrame
    k: int
        The numbers of neighbor

    Returns
    -------
    k3nerror : float
        k3n-Z-error or k3n-X-error
    """
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_dist = distance.cdist(X1, X1)
    X1_sorted_indices = np.argsort(X1_dist, axis=1)
    X2_dist = distance.cdist(X2, X2)

    for i in range(X2.shape[0]):
        _replace_zero_with_the_smallest_positive_values(X2_dist[i, :])

    I = np.eye(len(X1_dist), dtype=bool)
    neighbor_dist_in_X1 = np.sort(X2_dist[:, X1_sorted_indices[:, 1:k+1]][I])
    neighbor_dist_in_X2 = np.sort(X2_dist)[:, 1:k+1]

    sum_k3nerror = (
            (neighbor_dist_in_X1 - neighbor_dist_in_X2) / neighbor_dist_in_X2
           ).sum()
    return sum_k3nerror / X1.shape[0] / k


def _replace_zero_with_the_smallest_positive_values(arr):
    """
    Replace zeros in array with the smallest positive values.

    Parameters
    ----------
    arr: numpy.array
    """
    arr[arr==0] = np.min(arr[arr!=0])

