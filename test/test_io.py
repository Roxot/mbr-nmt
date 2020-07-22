import unittest

from mbr_nmt.io import read_candidates_file

class TestIO(unittest.TestCase):

    def test_read_candidates(self):
        candidates = read_candidates_file("test/data/candidates-3.txt", 3)
        expected = [["George went to school by bike today .".split(" "), 
                     "Today , George went to school by bike .".split(" "),
                     "This is a completely unrelated sentence .".split(" ")],
                    ["He got the bike for his birthday .".split(" "),
                     "He got the bike for his anniversary .".split(" "),
                     "The birthday bike .".split(" ")]]
        self.assertEqual(candidates, expected)

        raised_exception = False
        try:
            candidates = read_candidates_file("test/data/candidates-3.txt", 4)
        except:
            raised_exception = True
        self.assertTrue(raised_exception)
