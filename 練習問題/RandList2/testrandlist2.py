import unittest
from randlist2 import randlist

class Test_testRandList2(unittest.TestCase):
    def test_randlist2(self):
        tmp   = randlist(10)
        self.assertEqual(10, len(tmp))
        inner = [x for x in tmp if 0 <= x <= 100]
        self.assertEqual(10, len(inner))

    def test_randlist2_upper(self):
        tmp   = randlist(10, upper=50)
        self.assertEqual(10, len(tmp))
        inner = [x for x in tmp if 0 <= x <= 50]
        self.assertEqual(10, len(inner))

    def test_randlist2_lower(self):
        tmp   = randlist(10, lower=50)
        self.assertEqual(10, len(tmp))
        inner = [x for x in tmp if 50 <= x <= 100]
        self.assertEqual(10, len(inner))

    def test_randlist2_upper_lower(self):
        tmp   = randlist(10, lower=50, upper=85)
        self.assertEqual(10, len(tmp))
        inner = [x for x in tmp if 50 <= x <= 85]
        self.assertEqual(10, len(inner))
        
        

if __name__ == '__main__':
    unittest.main()
