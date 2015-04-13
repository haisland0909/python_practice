import unittest, prob22

class Test_testprob22(unittest.TestCase):
    def test_make_harris_point_with_gaussian(self):
        obj = prob22.Prob22()
        obj.make_harris_point_with_gaussian(1)
        obj.plot_harris_points(name="sd_1.jpg")
        obj = prob22.Prob22()
        obj.make_harris_point_with_gaussian(5)
        obj.plot_harris_points(name="sd_5.jpg")
        obj = prob22.Prob22()
        obj.make_harris_point_with_gaussian(10)
        obj.plot_harris_points(name="sd_10.jpg")
        obj = prob22.Prob22()
        obj.make_harris_point_with_gaussian(20)
        obj.plot_harris_points(name="sd_20.jpg") 

if __name__ == '__main__':
    unittest.main()
