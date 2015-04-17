import unittest, unsupervisedlearningsample

class Test_testunsupervisedlearningsample(unittest.TestCase):
    def test_plot_iris_k_mean(self):
        obj = unsupervisedlearningsample.KMeansSample()
        obj.plot_iris_k_mean()

    def test_plot_lena_k_mean(self):
        obj = unsupervisedlearningsample.KMeansSample()
        obj.plot_lena_k_mean()

    def test_plot_ward_cluster_lena(self):
        obj = unsupervisedlearningsample.KMeansSample()
        obj.plot_ward_cluster_lena()

if __name__ == '__main__':
    unittest.main()
