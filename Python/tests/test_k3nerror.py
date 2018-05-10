# coding: utf-8
import unittest

from sklearn.datasets import load_iris

from gtm import gtm
from k3nerror import k3nerror


class K3NErrorTestCase(unittest.TestCase):

    def setUp(self):
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

    def test_k3nerror_with_gtm(self):
        k = 10
        actual = k3nerror(self.X, self.means, k)
        expected = 0.8332434958
        self.assertAlmostEqual(expected, actual, places=9)

    def get_k3nerror_calculation_speed(self):
        k = 10
        k3nerror(self.X, self.means, k)


if __name__ == '__main__':
    unittest.main()

