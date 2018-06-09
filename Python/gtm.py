# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# GTM (generative topographic mapping) class
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA


class gtm:

    def __init__(self, shape_of_map=[30, 30], shape_of_rbf_centers=[10, 10],
                 variance_of_rbfs=4, lambda_in_em_algorithm=0.001,
                 number_of_iterations=200, display_flag=1):
        self.shape_of_map = shape_of_map
        self.shape_of_rbf_centers = shape_of_rbf_centers
        self.variance_of_rbfs = variance_of_rbfs
        self.lambda_in_em_algorithm = lambda_in_em_algorithm
        self.number_of_iterations = number_of_iterations
        self.display_flag = display_flag

    def calculate_grids(self, num_x, num_y):
        """
        Calculate grid coordinates on the GTM map
        
        Parameters
        ----------
        num_x : int
            number_of_x_grids
        num_y : int
            number_of_y_grids
        """
        grids_x, grids_y = np.meshgrid(np.arange(0.0, num_x), np.arange(0.0, num_y))
        grids = np.c_[np.ndarray.flatten(grids_x)[:, np.newaxis],
                      np.ndarray.flatten(grids_y)[:, np.newaxis]]
        max_grids = grids.max(axis=0)
        grids[:, 0] = 2 * (grids[:, 0] - max_grids[0] / 2) / max_grids[0]
        grids[:, 1] = 2 * (grids[:, 1] - max_grids[1] / 2) / max_grids[1]
        return grids

    def fit(self, input_dataset):
        """
        Train the GTM map
                
        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.
        """
        input_dataset = np.array(input_dataset)
        self.success_flag = True
        # make rbf grids
        self.rbf_grids = self.calculate_grids(self.shape_of_rbf_centers[0],
                                              self.shape_of_rbf_centers[1])

        # make map grids
        self.map_grids = self.calculate_grids(self.shape_of_map[0],
                                              self.shape_of_map[1])

        # calculate phi of map_grids and rbf_grids
        distance_between_map_and_rbf_grids = cdist(self.map_grids, self.rbf_grids,
                                                   'sqeuclidean')
        self.phi_of_map_rbf_grids = np.exp(-distance_between_map_and_rbf_grids / 2.0
                                           / self.variance_of_rbfs)

        # PCA for initializing W and beta
        pca_model = PCA(n_components=3)
        pca_model.fit_transform(input_dataset)
        if np.linalg.matrix_rank(self.phi_of_map_rbf_grids) < min(self.phi_of_map_rbf_grids.shape):
            self.success_flag = False
            return
        self.W = np.linalg.pinv(self.phi_of_map_rbf_grids).dot(
            self.map_grids.dot(pca_model.components_[0:2, :]))
        self.beta = min(pca_model.explained_variance_[2], 1 / (
                (
                        cdist(self.phi_of_map_rbf_grids.dot(self.W),
                              self.phi_of_map_rbf_grids.dot(self.W))
                        + np.diag(np.ones(np.prod(self.shape_of_map)) * 10 ** 100)
                ).min(axis=0).mean() / 2))
        self.bias = input_dataset.mean(axis=0)

        # EM algorithm
        phi_of_map_rbf_grids_with_one = np.c_[self.phi_of_map_rbf_grids,
                                              np.ones((np.prod(self.shape_of_map), 1))]
        for iteration in range(self.number_of_iterations):
            responsibilities = self.responsibility(input_dataset)

            phi_t_G_phi_etc = phi_of_map_rbf_grids_with_one.T.dot(
                np.diag(responsibilities.sum(axis=0)).dot(phi_of_map_rbf_grids_with_one)
            ) + self.lambda_in_em_algorithm / self.beta * np.identity(
                phi_of_map_rbf_grids_with_one.shape[1])
            if 1 / np.linalg.cond(phi_t_G_phi_etc) < 10 ** -15:
                self.success_flag = False
                break
            self.W_with_one = np.linalg.inv(phi_t_G_phi_etc).dot(
                phi_of_map_rbf_grids_with_one.T.dot(responsibilities.T.dot(input_dataset)))
            self.beta = input_dataset.size / (responsibilities
                                              * cdist(input_dataset,
                                                      phi_of_map_rbf_grids_with_one.dot(self.W_with_one)) ** 2).sum()

            self.W = self.W_with_one[:-1, :]
            self.bias = self.W_with_one[-1, :]

            if self.display_flag:
                print("{0}/{1} ... likelihood: {2}".format(iteration + 1, self.number_of_iterations,
                                                           self.likelihood_value))

    def calculate_distance_between_phi_w_and_input_distances(self, input_dataset):
        """
        Calculate distance between phi*W
        
        Parameters
        ----------
        input_dataset : numpy.array
             Training dataset for GTM.
             
        Returns
        -------
        distance : distance between phi*W
        """
        distance = cdist(
            input_dataset,
            self.phi_of_map_rbf_grids.dot(self.W)
            + np.ones((np.prod(self.shape_of_map), 1)).dot(
                np.reshape(self.bias, (1, len(self.bias)))
            ),
            'sqeuclidean')
        return distance

    def responsibility(self, input_dataset):
        """
        Get responsibilities and likelihood.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.

        Returns
        -------
        reponsibilities : numpy.array
            Responsibilities of input_dataset for each grid point.
        likelihood_value : float
            likelihood of input_dataset.
        """
        input_dataset = np.array(input_dataset)
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)
        rbf_for_responsibility = np.exp(-self.beta / 2.0 * distance)
        sum_of_rbf_for_responsibility = rbf_for_responsibility.sum(axis=1)
        zero_sample_index = np.where(sum_of_rbf_for_responsibility == 0)[0]
        if len(zero_sample_index):
            sum_of_rbf_for_responsibility[zero_sample_index] = 1
            rbf_for_responsibility[zero_sample_index, :] = 1 / rbf_for_responsibility.shape[1]
        
        reponsibilities = rbf_for_responsibility / np.reshape(sum_of_rbf_for_responsibility,
                                                              (rbf_for_responsibility.shape[0], 1))

        self.likelihood_value = (np.log((self.beta / 2.0 / np.pi) ** (input_dataset.shape[1] / 2.0) /
                                        np.prod(self.shape_of_map) * rbf_for_responsibility.sum(axis=1))).sum()

        return reponsibilities

    def likelihood(self, input_dataset):
        """
        Get likelihood.

        Parameters
        ----------
        input_dataset : numpy.array or pandas.DataFrame
             Training dataset for GTM.
             input_dataset must be autoscaled.

        Returns
        -------
        likelihood : scalar
            likelihood of input_dataset.
        """
        input_dataset = np.array(input_dataset)
        distance = self.calculate_distance_between_phi_w_and_input_distances(input_dataset)
        return (np.log((self.beta / 2.0 / np.pi) ** (input_dataset.shape[1] / 2.0) /
                       np.prod(self.shape_of_map) * np.exp(-self.beta / 2.0 * distance).sum(axis=1))).sum()

    def mlr(self, X, y):
        """
        Train the MLR model
        
        Parameters
        ----------
        X, y : numpy.array or pandas.DataFrame
            Both X and y must NOT be autoscaled.
        """
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))
        # autoscaling
        self.Xmean = X.mean(axis=0)
        self.Xstd = X.std(axis=0, ddof=1)
        autoscaled_X = (X - self.Xmean) / self.Xstd
        self.y_mean = y.mean(axis=0)
        self.ystd = y.std(axis=0, ddof=1)
        autoscaled_y = (y - self.y_mean) / self.ystd
        self.regression_coefficients = np.linalg.inv(
            np.dot(autoscaled_X.T, autoscaled_X)
        ).dot(autoscaled_X.T.dot(autoscaled_y))
        calculated_y = (autoscaled_X.dot(self.regression_coefficients)
                        * self.ystd + self.y_mean)
        self.sigma = sum((y - calculated_y) ** 2) / len(y)

    def mlr_predict(self, X):
        """
        Predict y-values from X-values using the MLR model
        
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            X must NOT be autoscaled.
        """
        autoscaled_X = (X - self.Xmean) / self.Xstd
        return (autoscaled_X.dot(self.regression_coefficients)
                * self.ystd + self.y_mean)

    def inverse_gtm_mlr(self, target_y_value):
        """
        Predict X-values from a y-value using the MLR model
        
        Parameters
        ----------
        target_v_alue : a target y-value
            scaler

        Returns
        -------
        responsibilities_inverse can be used to discussed assigned grids on
        the GTM map.
        """
        #        target_y_values = np.ndarray.flatten(np.array(target_y_values))
        myu_i = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
            (np.prod(self.shape_of_map), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))
        sigma_i = np.diag(np.ones(len(self.regression_coefficients))) / self.beta
        inverse_sigma_i = np.diag(np.ones(len(self.regression_coefficients))) * self.beta
        delta_i = np.linalg.inv(inverse_sigma_i
                                + self.regression_coefficients.dot(self.regression_coefficients.T) / self.sigma)
        #        for target_y_value in target_y_values:
        pxy_means = np.empty(myu_i.shape)
        for i in range(pxy_means.shape[0]):
            pxy_means[i, :] = np.ndarray.flatten(
                delta_i.dot(
                    self.regression_coefficients / self.sigma * target_y_value
                    + inverse_sigma_i.dot(np.reshape(myu_i[i, :], [myu_i.shape[1], 1]))
                ))

        pyz_means = myu_i.dot(self.regression_coefficients)
        pyz_var = self.sigma + self.regression_coefficients.T.dot(
            sigma_i.dot(self.regression_coefficients))
        pyzs = np.empty(len(pyz_means))
        for i in range(len(pyz_means)):
            pyzs[i] = norm.pdf(target_y_value, pyz_means[i], pyz_var ** (1 / 2))

        responsibilities_inverse = pyzs / pyzs.sum()
        estimated_x_mean = responsibilities_inverse.dot(pxy_means)
        estimated_x_mode = pxy_means[np.argmax(responsibilities_inverse), :]

        # pyzs : vector of probability of y given zi, which can be used to
        #        discuss applicability domains
        return estimated_x_mean, estimated_x_mode, responsibilities_inverse

    def gtmr_predict(self, X):
        """
        Predict y-values from X-values using the GTMR model
        
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            X must be autoscaled.
        Multiple y-variables are OK.
        In model, the rigth p variables are handled as y-variables ( p is the
        number of y-variables ).

        Returns
        responsibilities can be used to discussed assigned grids on the GTM
        map.
        px [p(x)] : vector of probability of x given myu_x_i and sigma_x_i,
        which can be used to discuss applicability domains.
        -------

        """
        if self.success_flag:
            X = np.array(X)
            myu_i = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
                (np.prod(self.shape_of_map), 1)
            ).dot(np.reshape(self.bias, (1, len(self.bias))))
            delta_x = np.diag(np.ones(X.shape[1])) / self.beta
            px = np.empty([X.shape[0], myu_i.shape[0]])
            for i in range(myu_i.shape[0]):
                px[:, i] = multivariate_normal.pdf(X, myu_i[i, 0:X.shape[1]], delta_x)

            responsibilities = px.T / px.T.sum(axis=0)
            responsibilities = responsibilities.T
            estimated_y_mean = responsibilities.dot(myu_i[:, X.shape[1]:])
            estimated_y_mode = myu_i[np.argmax(responsibilities, axis=1), X.shape[1]:]
        else:
            estimated_y_mean = np.zeros(X.spape[0])
            estimated_y_mode = np.zeros(X.spape[0])
            px = np.empty([X.shape[0], np.prod(self.shape_of_map)])
            responsibilities = np.empty([X.shape[0], np.prod(self.shape_of_map)])

        return estimated_y_mean, estimated_y_mode, responsibilities, px

    def inverse_gtmr(self, target_y_value):
        """
        Predict X-values from y-values using the GTMR model
        
        Parameters
        ----------
        targe_y_value must be one candidate. But, multiple y-variables are OK.
        In model, the rigth m variables are handled as X-variables ( m is the
        number of X-variables ).

        Returns
        -------
        responsibilities_inverse can be used to discussed assigned grids on
        the GTM map.
        py [p(y)] : vector of probability of y given myu_y_i and sigma_y_i,
        which can be used to discuss applicability domains.
        """
        myu_i = self.phi_of_map_rbf_grids.dot(self.W) + np.ones(
            (np.prod(self.shape_of_map), 1)).dot(np.reshape(self.bias, (1, len(self.bias))))
        delta_y = 1 / self.beta
        py = np.empty(myu_i.shape[0])
        if isinstance(target_y_value, int) or isinstance(target_y_value, float):
            for i in range(myu_i.shape[0]):
                py[i] = multivariate_normal.pdf(target_y_value, myu_i[i, -1], delta_y)

            responsibilities_inverse = py / py.sum()
            estimated_x_mean = responsibilities_inverse.dot(myu_i[:, 0:-1])
            estimated_x_mode = myu_i[np.argmax(responsibilities_inverse), 0:-1]
        else:
            for i in range(myu_i.shape[0]):
                py[i] = multivariate_normal.pdf(
                    target_y_value, myu_i[i, -len(target_y_value)], delta_y)

            responsibilities_inverse = py / py.sum()
            estimated_x_mean = responsibilities_inverse.dot(myu_i[:, 0:-len(target_y_value)])
            estimated_x_mode = myu_i[np.argmax(responsibilities_inverse), 0:-len(target_y_value)]

        return estimated_x_mean, estimated_x_mode, responsibilities_inverse, py
