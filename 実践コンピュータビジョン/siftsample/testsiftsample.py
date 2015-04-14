import unittest, siftsample

class Test_testsiftsample(unittest.TestCase):
    def test_make_sift_feature(self):
        obj = siftsample.SiftSample()
        obj.make_sift_feature()
        self.assertEqual(1090.9000000000001, obj.get_shift_location()[0][0])
        self.assertEqual(0, obj.get_sift_descriptors()[0][0])

    def test_plot_sift_feature(self):
        obj = siftsample.SiftSample()
        obj.save("base_image.jpg")
        obj.plot_sift_feature("circle_sift_point.jpg", True)
        obj.plot_sift_feature("normal_sift_point.jpg")

    def test_sift_match(self):
        obj1 = siftsample.SiftSample()
        obj2 = siftsample.SiftSample("cut_image.jpg")
        obj  = siftsample.SiftMatch(obj1, obj2)
        obj.match()
        self.assertEqual(0, obj.get_match_score()[0])

    def test_plot_matches(self):
        obj1 = siftsample.SiftSample()
        obj2 = siftsample.SiftSample("reverse_image.jpg")
        obj  = siftsample.SiftMatch(obj1, obj2)
        obj.plot_matches("sift_match.jpg")

if __name__ == '__main__':
    unittest.main()
