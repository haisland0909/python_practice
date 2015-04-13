import unittest, prob14

class Test_testprob14(unittest.TestCase):
    def test_create_edge(self):
        obj = prob14.Prob14("black_square.jpg")
        obj.save("base_image.jpg")
        im  = obj.create_edge()
        im.save("square_edge.jpg")

if __name__ == '__main__':
    unittest.main()
