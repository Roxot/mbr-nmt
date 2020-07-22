import unittest

from mbr_nmt.utility import unigram_precision

class TestUtility(unittest.TestCase):

    def test_unigram_precision(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")
        hyp4 = "a a a a a a".split(" ")
        hyps = [hyp1, hyp2, hyp3, hyp4]

        expected = 1.
        for hyp in hyps:
            result = unigram_precision(hyp, hyp)
            self.assertEqual(result, expected)

        result = unigram_precision(hyp1, hyp2)
        expected = 7.0 / 8.0
        self.assertEqual(result, expected)

        result = unigram_precision(hyp1, hyp3)
        expected = 1.0 / 8.0
        self.assertEqual(result, expected)

        result = unigram_precision(hyp1, hyp4)
        expected = 0.
        self.assertEqual(result, expected)
