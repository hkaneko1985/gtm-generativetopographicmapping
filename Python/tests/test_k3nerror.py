# coding: utf-8
import unittest

from sklearn.datasets import load_boston, load_iris

from gtm import gtm
from k3nerror import k3nerror


class K3NErrorTestCase(unittest.TestCase):

    def test_k3nerror_with_gtm_to_iris(self):
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

        k = 10
        actual = k3nerror(X, means, k)
        expected = 0.8332434959
        self.assertAlmostEqual(expected, actual, places=7)

    def test_k3nerror_with_gtm_to_boston(self):
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

        k = 10
        actual = k3nerror(X, means, k)
        expected = 13.53674187
        self.assertAlmostEqual(expected, actual, places=7)


class K3NErrorRunSpeedTest:

    def setup_to_boston(self):
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambdas = 0.001
        iterations = 300
        do_display = 0

        boston = load_boston()
        self.X = boston.data

        #autoscaling
        self.X = (self.X-self.X.mean(axis=0)) / self.X.std(axis=0,ddof=1)

        model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs,
                    lambdas, iterations, do_display)
        model.fit(self.X)

        responsibilities = model.responsibility(self.X)
        self.means = responsibilities.dot(model.mapgrids)

    def setup_to_iris(self):
        shape_of_map = [10, 10]
        shape_of_rbf_centers = [5, 5]
        variance_of_rbfs = 4
        lambdas = 0.001
        iterations = 300
        do_display = 0

        iris = load_iris()
        self.X = iris.data

        #autoscaling
        self.X = (self.X-self.X.mean(axis=0)) / self.X.std(axis=0,ddof=1)

        model = gtm(shape_of_map, shape_of_rbf_centers, variance_of_rbfs,
                    lambdas, iterations, do_display)
        model.fit(self.X)

        responsibilities = model.responsibility(self.X)
        self.means = responsibilities.dot(model.mapgrids)

    def get_k3nerror_calculation_speed_with_k_equals(self, k):
        k3nerror(self.X, self.means, k)


if __name__ == '__main__':
    unittest.main()

