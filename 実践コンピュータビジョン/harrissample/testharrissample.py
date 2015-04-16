import unittest, harrissample, numpy

class Test_testharrissample(unittest.TestCase):
    def test_compute_harris_response(self):
        obj = harrissample.HarrisSample()
        obj.compute_harris_response(sigma = 1)
        self.assertEqual(0.15598916208901456,  obj.get_harris_response()[0,0])

    def test_make_harris_points(self):
        obj = harrissample.HarrisSample()
        obj.make_harris_points()
        self.assertEqual(265, obj.get_harris_point()[0][0])

    def test_plot_harris_points(self):
        obj = harrissample.HarrisSample()
        obj.save("base_image.jpg")
        obj.plot_harris_points("harris_point.jpg")
        obj = harrissample.HarrisSample("black_square.jpg")
        obj.save("square_image.jpg")
        obj.plot_harris_points("harris_point_square.jpg")

    def test_calc_descriptors(self):
        obj = harrissample.HarrisSample()
        obj.calc_descriptors()
        self.assertEqual(27, obj.get_descriptors()[0][0])

    def test_harris_match(self):
        obj1 = harrissample.HarrisSample()
        obj2 = harrissample.HarrisSample("cut_image.jpg")
        obj  = harrissample.HarrisMatch(obj1, obj2)
        obj.match()
        self.assertEqual(486, obj.get_match_score()[0])

    def test_appendimages(self):
        obj1 = harrissample.HarrisSample()
        obj2 = harrissample.HarrisSample("cut_image.jpg")
        obj  = harrissample.HarrisMatch(obj1, obj2)
        obj.appendimages()
        self.assertEqual((960, 2560, 3), obj.get_append_image().shape)
        obj.save_append_image("append_image.jpg")

    def test_plot_matches(self):
        obj1 = harrissample.HarrisSample()
        obj2 = harrissample.HarrisSample("cut_image.jpg")
        obj  = harrissample.HarrisMatch(obj1, obj2)
        obj.plot_matches()
        obj.plot_matches(name = "harris_match_20.jpg", match_maximum = 20)
        
if __name__ == '__main__':
    unittest.main()
