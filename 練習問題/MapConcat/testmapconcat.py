import unittest
from mapconcat import mapconcat

class Test_testMapConcat(unittest.TestCase):
    def test_mapconcat(self):
        self.assertEqual("foo-bar-baz", mapconcat(str, ["foo", "bar", "baz"], "-")) 
        self.assertEqual("1 2 3", mapconcat(str, [1, 2, 3], " ")) 
        self.assertEqual("aaabbbccc", mapconcat(lambda c: c*3, "abc", "")) 
        self.assertEqual("       foo       bar       baz", mapconcat(lambda s: s.rjust(10), ["foo", "bar", "baz"], "")) 
        
        

if __name__ == '__main__':
    unittest.main()
