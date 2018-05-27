# coding: utf-8
import unittest

import numpy as np
from sklearn.datasets import load_boston, load_iris

from gtm import gtm
from k3nerror import k3nerror


class GTMTestCase(unittest.TestCase):

    def test_gtm_with_iris(self):
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambdas = 0.001
        iterations = 300
        do_display = 0

        iris = load_iris()
        X = iris.data

        #autoscaling
        X = (X-X.mean(axis=0)) / X.std(axis=0,ddof=1)

        model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs,
                    lambdas, iterations, do_display)
        model.fit(X)

        responsibilities = model.responsibility(X)
        means = responsibilities.dot(model.mapgrids)
        actual = means[:10, :]
        expected = np.array([
                    [-0.6705465933468514,0.3829717245213705],
                    [-0.48316715705696234,0.48573876891327505],
                    [-0.9326499402098388,-0.4653586650861626],
                    [-0.8796540759427155,-0.432748196248535],
                    [-0.808484188863981,0.1127914668082637],
                    [-0.8562671930913747,0.5054770179076009],
                    [-0.8827856441300941,-0.1900002396932689],
                    [-0.6567763264305273,0.43100475126529186],
                    [-0.9675496106656379,-0.7475745364276133],
                    [-0.6369141396343855,0.15331616956496824]
                   ])
        np.testing.assert_allclose(expected, actual, rtol=1e-7, atol=0)

        k = 10
        actual = k3nerror(X, means, k)
        expected = 0.8332434959
        self.assertAlmostEqual(expected, actual, places=7)

    def test_gtm_with_boston(self):
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambdas = 0.001
        iterations = 300
        do_display = 0

        boston = load_boston()
        X = boston.data

        #autoscaling
        X = (X-X.mean(axis=0)) / X.std(axis=0,ddof=1)

        model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs,
                    lambdas, iterations, do_display)
        model.fit(X)

        responsibilities = model.responsibility(X)
        means = responsibilities.dot(model.mapgrids)
        actual = means[:10, :]
        expected = np.array([
                    [-0.9121181133961582,0.3270687429716633],
                    [-0.7623993487555986,-0.03465058258516577],
                    [-0.9751389665503443,0.47031838071377396],
                    [-0.9881881283635213,0.17674611687624858],
                    [-0.9834545901596559,0.2612741266073518],
                    [-0.9416833225443894,0.06562281187680431],
                    [-0.5371087450312528,-0.8834321955958397],
                    [-0.39080236753376996,-0.9576518330286727],
                    [-0.32352324436445645,-0.9372201378955927],
                    [-0.4773446121858353,-0.9178463400391587],
                  ])
        np.testing.assert_allclose(expected, actual, rtol=1e-7, atol=0)

        k = 10
        actual = k3nerror(X, means, k)
        expected = 13.53674187
        self.assertAlmostEqual(expected, actual, places=7)


class GTMRunSpeedTest:

    def setup_to_boston(self):
        boston = load_boston()
        self.X = boston.data
        #autoscaling
        self.X = (self.X-self.X.mean(axis=0)) / self.X.std(axis=0,ddof=1)

    def setup_to_iris(self):
        iris = load_iris()
        self.X = iris.data
        #autoscaling
        self.X = (self.X-self.X.mean(axis=0)) / self.X.std(axis=0,ddof=1)

    def get_k3nerror_calculation_speed_with_k_equals(self, k):
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambdas = 0.001
        iterations = 300
        do_display = 0

        model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs,
                    lambdas, iterations, do_display)
        model.fit(self.X)

        responsibilities = model.responsibility(self.X)
        means = responsibilities.dot(model.mapgrids)


if __name__ == '__main__':
    unittest.main()

