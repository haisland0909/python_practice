import sys
sys.path.append('../harrissample')
import unittest, prob21, harrissample


class Test_testprob21(unittest.TestCase):
    def test_harris_match_21(self):
        obj1 = harrissample.HarrisSample()
        obj2 = harrissample.HarrisSample("cut_image.jpg")
        obj  = prob21.Prob21(obj1, obj2)
        obj.harris_match(max_rel = 0.75)
        obj.plot_matches()
        obj.plot_matches(name = "harris_match_20.jpg", match_maximum = 20)


if __name__ == '__main__':
    unittest.main()
