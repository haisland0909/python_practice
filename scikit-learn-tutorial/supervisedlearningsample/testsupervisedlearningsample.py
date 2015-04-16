import unittest, supervisedlearningsample
from sklearn import linear_model

class Test_testsupervisedlearningsample(unittest.TestCase):
    def test_plot_iris_data(self):
        obj = supervisedlearningsample.KNearestSample()
        obj.plot_iris_data()

    def test_do_k_nearest(self):
        obj = supervisedlearningsample.KNearestSample()
        obj.do_k_nearest(1)
        obj.do_k_nearest(5)
        obj.do_k_nearest(10)

    def test_do_linear_regression(self):
        obj = supervisedlearningsample.LinearSample()
        obj.do_linear_regression()

    def test_do_linear_few_data(self):
        obj = supervisedlearningsample.LinearSample()
        obj.do_linear_few_data()
        obj.do_linear_few_data(linear_model.Ridge(alpha=.1))


if __name__ == '__main__':
    unittest.main()
