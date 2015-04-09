import unittest, pilsample

class Test_testpilsample(unittest.TestCase):
    def test_get_image(self):
        obj = pilsample.PilSample()
        im  = obj.get_image()
        self.assertEqual("JPEG", im.format)
        self.assertEqual((1280, 960), im.size)
        self.assertEqual("RGB", im.mode)

if __name__ == '__main__':
    unittest.main()
