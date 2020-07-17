import unittest

from mbr_nmt.io import read_candidates_file

class TestIO(unittest.TestCase):

    def test_read_candidates(self):
        candidates = read_candidates_file("test/data/candidates-3.txt", 3)
        expected = [["George went to school by bike today.", "Today, George went to school by bike.",  "This is a completely unrelated sentence."],
                    ["He got the bike for his birthday.", "He got the bike for his anniversary.",  "The birthday bike."]]
        self.assertEqual(candidates, expected)

        raised_exception = False
        try:
            candidates = read_candidates_file("test/data/candidates-3.txt", 4)
        except:
            raised_exception = True
        self.assertTrue(raised_exception)
