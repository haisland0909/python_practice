import unittest, scipysample, numpy

class Test_testscipysample(unittest.TestCase):
    def test_get_grad_x(self):
        obj = scipysample.ScipySample()
        g   = obj.get_grad_x()
        self.assertEqual((960, 1280, 3), g.shape)
        obj.convert_grey()
        g   = obj.get_grad_x()
        self.assertEqual((960, 1280), g.shape)

    def test_get_grad_y(self):
        obj = scipysample.ScipySample()
        g   = obj.get_grad_y()
        self.assertEqual((960, 1280, 3), g.shape)
        obj.convert_grey()
        g   = obj.get_grad_y()
        self.assertEqual((960, 1280), g.shape)

    def test_get_grad_mag(self):
        obj = scipysample.ScipySample()
        g   = obj.get_grad_mag()
        self.assertEqual((960, 1280, 3), g.shape)
        obj.convert_grey()
        g   = obj.get_grad_mag()
        self.assertEqual((960, 1280), g.shape)

    def test_calc_gradiation(self):
        obj = scipysample.ScipySample()
        obj.calc_gradiation(1)
        x   = obj._image_grad_x
        y   = obj._image_grad_y
        mag = obj._image_grad_mag
        self.assertEqual(mag.all(), numpy.sqrt(x ** 2 + y ** 2).all())

if __name__ == '__main__':
    unittest.main()
