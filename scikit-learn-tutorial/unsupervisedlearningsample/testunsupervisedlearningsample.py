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

    def test_plot_pca(self):
        obj = unsupervisedlearningsample.DecompositionsSample()
        obj.plot_pca()

    def test_plot_ica(self):
        obj = unsupervisedlearningsample.DecompositionsSample()
        obj.plot_ica()

if __name__ == '__main__':
    unittest.main()
