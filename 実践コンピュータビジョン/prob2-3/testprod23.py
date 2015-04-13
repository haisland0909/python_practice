import unittest, prod23

class Test_prod23(unittest.TestCase):
    def test_make_fast_points(self):
        obj = prod23.Prob23()
        obj.make_fast_points()
        self.assertEqual(271, obj.get_fast_points()[0][0])

    def test_plot_fast_points(self):
        obj = prod23.Prob23()
        obj.save("base_image.jpg")
        obj.plot_fast_points("fast_point.jpg")
        obj.make_fast_points(threshold = 40)
        obj.plot_fast_points("fast_point_40.jpg")

if __name__ == '__main__':
    unittest.main()
