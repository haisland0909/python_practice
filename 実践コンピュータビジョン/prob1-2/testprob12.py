import unittest, prob12
class Test_testprob12(unittest.TestCase):
    def test_add_unsharped_mask(self):
        obj   = prob12.Prob12()
        sd_1  = obj.add_unsharped_mask(1)
        obj.save('sd_1.jpg')
        obj   = prob12.Prob12()
        sd_1  = obj.add_unsharped_mask(5)
        obj.save('sd_5.jpg')
        obj   = prob12.Prob12()
        sd_1  = obj.add_unsharped_mask(10)
        obj.save('sd_10.jpg')
        

if __name__ == '__main__':
    unittest.main()