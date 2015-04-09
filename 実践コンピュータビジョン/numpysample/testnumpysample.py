import unittest, numpysample

class Test_testnumpysample(unittest.TestCase):
    def test_get_array_image(self):
        obj = numpysample.NumpySample()
        im  = obj.get_array_image()
        self.assertEqual((960, 1280, 3), im.shape)
        obj.convert_grey()
        im  = obj.get_array_image()
        self.assertEqual((960, 1280), im.shape)
       

    def test_change_color_level(self):
        obj        = numpysample.NumpySample()
        im         = obj.get_array_image()[0, 0, 0]
        obj.change_color_level(lambda x: 255 - x)
        im_reverse = obj.get_array_image()[0, 0, 0]
        self.assertEqual(255 - im, im_reverse)
        obj.save('reverse.jpg')

    def test_resize_image(self):
        obj      = numpysample.NumpySample()
        obj.resize_image((640, 480))
        im_small = obj.get_array_image()
        self.assertEqual((480, 640, 3), im_small.shape)
        obj.save("small.jpg")


if __name__ == '__main__':
    unittest.main()
