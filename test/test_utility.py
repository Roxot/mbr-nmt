import unittest

from mbr_nmt.utility import unigram_precision, BEER, METEOR, BLEU, ChrF, TER

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
    
    def test_bleu(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")
        hyp4 = "a a a a a a".split(" ")
        hyps = [hyp1, hyp2, hyp3, hyp4]

        bleu = BLEU()
        expected = 1.
        for hyp in hyps:
            result = bleu(hyp, hyp)
            self.assertEqual(result, expected)

        for ref, expected in zip(hyps, [100., 62.40, 0.0, 0.0]):
            result = bleu(hyp1, ref)
            self.assertEqual(result, expected)
    
    def test_chrf(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")
        hyp4 = "a a a a a a".split(" ")
        hyps = [hyp1, hyp2, hyp3, hyp4]

        chrf = ChrF()
        expected = 1.
        for hyp in hyps:
            result = chrf(hyp, hyp)
            self.assertEqual(result, expected)

        for ref, expected in zip(hyps, [1., 0.8202, 0.1134, 0.0154]):
            result = chrf(hyp1, ref)
            self.assertEqual(result, expected)
    
    
    def test_ter(self):
        hyp1 = "George went to school by bike today .".split(" ")
        hyp2 = "Today , George went to school by bike .".split(" ")
        hyp3 = "This is a completely unrelated sentence .".split(" ")
        hyp4 = "a a a a a a".split(" ")
        hyps = [hyp1, hyp2, hyp3, hyp4]

        ter = TER()
        expected = 1.
        for hyp in hyps:
            result = ter(hyp, hyp)
            self.assertEqual(result, expected)

        for ref, expected in zip(hyps, [0.0, -0.2222, -1.0, -1.3333]):
            result = ter(hyp1, ref)
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

