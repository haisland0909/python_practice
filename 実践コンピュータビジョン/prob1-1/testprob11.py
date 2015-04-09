import unittest, prob11
class Test_testprob11(unittest.TestCase):
    def test_create_gaussian_contour(self):
        obj   = prob11.Prob11()
        sd_1  = obj.create_gaussian_contour(1, 'sd_1.jpg')
        sd_2  = obj.create_gaussian_contour(2, 'sd_2.jpg')
        sd_5  = obj.create_gaussian_contour(5, 'sd_5.jpg')
        sd_10 = obj.create_gaussian_contour(10, 'sd_10.jpg')
        

if __name__ == '__main__':
    unittest.main()
