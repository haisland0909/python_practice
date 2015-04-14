import unittest, prob25

class Test_testprob25(unittest.TestCase):
    def test_mser_plot_sift_feature(self):
        obj = prob25.Prob25()
        obj.plot_sift_feature("mser_point.jpg", True)

if __name__ == '__main__':
    unittest.main()
