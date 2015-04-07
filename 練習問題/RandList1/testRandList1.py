import unittest
from randlist1 import randlist

class Test_testRandList1(unittest.TestCase):
    def test_randlist(self):
        tmp   = randlist(10)
        self.assertEqual(10, len(tmp))
        inner = [x for x in tmp if 0 < x < 100]
        self.assertEqual(10, len(inner)) 

if __name__ == '__main__':
    unittest.main()
