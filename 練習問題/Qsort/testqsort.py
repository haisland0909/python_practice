import unittest
from qsort import qsort, slencmp

class Test_testRandList1(unittest.TestCase):
    def test_qsort(self):
        self.assertEqual([-90, 2, 3, 4, 10], qsort(lambda x, y: x < y, [2, 4, -90, 3, 10])) 
        self.assertEqual([10, 4, 3, 2, -90], qsort(lambda x, y: x > y, [2, 4, -90, 3, 10])) 

    def test_slencmp(self):
         self.assertEqual(['int', 'long', 'short', 'double'], qsort(slencmp, ["short", "double", "int", "long"]))

if __name__ == '__main__':
    unittest.main()