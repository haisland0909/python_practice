import sys
sys.path.append('../siftsample')
import unittest, siftsample

class Test_testprob2_4(unittest.TestCase):
    def test_prob24(self):
        obj1 = siftsample.SiftSample()
        obj2 = siftsample.SiftSample("small_image.jpg")
        obj  = siftsample.SiftMatch(obj1, obj2)
        obj.plot_matches("sift_match.jpg", match_maximum = 10)

if __name__ == '__main__':
    unittest.main()
