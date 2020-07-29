import unittest

from mbr_nmt.utility import unigram_precision, BEER, METEOR

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
 
    def test_beer(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")

        # It's difficult to test BEER, but we perform some sanity checks. 
        beer = BEER()
        score11 = beer(hyp1, hyp1)
        score12 = beer(hyp1, hyp2)
        score13 = beer(hyp1, hyp3)        
        self.assertTrue(isinstance(score12, float))
        self.assertTrue(score12 >= 0. and score12 <= 1.)
        self.assertTrue(score13 >= 0. and score13 <= 1.)
        self.assertTrue(score11 > score12 and score11 > score13)
        self.assertTrue(score12 > score13)

    def test_meteor(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")

        # Test that all available languages work.
        for lang in METEOR.available_languages:

            # Start the METEOR server, this might take a couple of seconds.
            meteor = METEOR(lang=lang)

            # Perform some sanity checks.
            score11 = meteor(hyp1, hyp1)
            score12 = meteor(hyp1, hyp2)
            score13 = meteor(hyp1, hyp3)        
            self.assertEqual(score11, 1.0)
            self.assertTrue(isinstance(score12, float))
            self.assertTrue(score12 >= 0. and score12 <= 1.)
            self.assertTrue(score13 >= 0. and score13 <= 1.)
            self.assertTrue(score12 > score13)
        
            # Close the METEOR server.
            del meteor

