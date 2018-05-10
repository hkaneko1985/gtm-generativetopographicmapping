# coding: utf-8
import numpy as np


def _calculate_distance(X, sample_number):
    return np.ndarray.flatten(
               np.sqrt(((X[:, np.newaxis]-X[sample_number, :])**2).sum(axis=2))
           )


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
    sum_of_k3nerror = 0
    X1 = np.array(X1)
    X2 = np.array(X2)
    for sample_number in range(X1.shape[0]):
        X1_distance = _calculate_distance(X1, sample_number)
        X1_sorted_index = np.delete(np.argsort(X1_distance), 0)

        X2_distance = _calculate_distance(X2, sample_number)
        X2_sorted_index = np.delete(np.argsort(X2_distance), 0)
        X2_distance[X2_distance==0] = np.min(X2_distance[X2_distance!=0])

        sum_of_k3nerror += (
                            (np.sort(X2_distance[X1_sorted_index[0:k]])
                            - X2_distance[X2_sorted_index[0:k]])
                            / X2_distance[X2_sorted_index[0:k]]
                           ).sum()

    return sum_of_k3nerror / X1.shape[0] / k

