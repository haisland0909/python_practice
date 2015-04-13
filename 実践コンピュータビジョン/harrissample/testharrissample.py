import unittest, harrissample

class Test_testharrissample(unittest.TestCase):
    def test_compute_harris_response(self):
        obj = harrissample.HarrisSample()
        obj.compute_harris_response(sigma = 1)
        self.assertEqual(0.15598916208901456,  obj.get_harris_reponse()[0,0])

if __name__ == '__main__':
    unittest.main()
