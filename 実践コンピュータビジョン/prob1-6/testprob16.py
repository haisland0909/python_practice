import unittest, prob16, pylabsample, numpy

class Test_testprob16(unittest.TestCase):
    def test_convert_binary(self):
        obj = prob16.Prob16()
        obj.convert_binary(128)
        arr = obj.get_array_image()
        self.assertEqual(255, arr[0, 0])

    def test_get_label(self):
        obj = prob16.Prob16("many_object.jpg")
        obj.convert_binary(128)
        obj.get_label()
        obj.save("label_image.jpg")
        pylabsample.create_hist(obj.get_array_image().flatten(), "size_num_hist.jpg")

    def test_do_binary_opening(self):
        obj = prob16.Prob16("many_object.jpg")
        obj.convert_binary(128)
        obj.get_label()
        obj = prob16.Prob16("many_object.jpg")
        obj.convert_binary(128)
        obj.do_binary_opening(numpy.ones((9, 9)), 5)
        obj.get_label()
        obj.save("label_image_open.jpg")
        pylabsample.create_hist(obj.get_array_image().flatten(), "size_num_hist_open.jpg")



if __name__ == '__main__':
    unittest.main()
