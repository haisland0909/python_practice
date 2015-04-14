import unittest, prob13
class Test_testprob13(unittest.TestCase):
    def test_make_self_quotient(self):
        obj   = prob13.Prob13()
        obj.convert_grey()
        obj.make_self_quotient(1)
        obj.save('sd_1.jpg')
        obj   = prob13.Prob13()
        obj.convert_grey()
        obj.make_self_quotient(5)
        obj.save('sd_5.jpg')
        obj   = prob13.Prob13()
        obj.convert_grey()
        obj.make_self_quotient(10)
        obj.save('sd_10.jpg')
        obj   = prob13.Prob13()
        obj.convert_grey()
        obj.make_self_quotient(20)
        obj.save('sd_20.jpg')
        

if __name__ == '__main__':
    unittest.main()
