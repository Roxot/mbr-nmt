import unittest
import numpy as np 

from mbr_nmt.mbr import mbr
from mbr_nmt.utility import NGramPrecision

class TestMBR(unittest.TestCase):

    def test_mbr(self):
        unigram_precision = NGramPrecision(1)
        samples = [["George went to school by bike today .",
                     "Today , George went to school by bike .",
                     "This is a completely unrelated sentence ."],
                   ["It was a red bike .",
                    "The bike was red .",
                    "The bike ."]]

        # Test matrix computations and prediction.
        expected_matrix = np.array([[1., 7./8., 1./8.],
                                    [7./9., 1., 1./9.],
                                    [1./7., 1./7., 1.]])
        expected_pred = samples[0][0]
        result_pred, result_matrix = mbr(samples[0], unigram_precision, candidates=None, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # One more test on another set of sentences.
        expected_matrix = np.array([[1., 4./6., 2./6.],
                                    [4./5., 1., 3./5.],
                                    [2./3., 3./3., 1.]])
        expected_pred = samples[1][2]
        result_pred, result_matrix = mbr(samples[1], unigram_precision, candidates=None, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # Test that separate candidates and samples work.
        candidates = ["It was a red bike .",
                      "The bike was red .",
                      "The bike .",
                      "bike ."]
        samples = ["It was a red bike .",
                   "The bike was red .",
                   "The bike ."]
        expected_matrix = np.array([[1., 4./6., 2./6.],
                                    [4./5., 1., 3./5.],
                                    [2./3., 3./3., 1.],
                                    [2./2., 2./2., 2./2.]])
        expected_pred = candidates[3]
        result_pred, result_matrix = mbr(samples, unigram_precision, candidates=candidates, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # And when we have more samples than candidates.
        candidates = ["George went to school by bike today .", 
                      "Today , George went to school by bike ."]
        samples = ["George went to school by bike today .", 
                   "Today , George went to school by bike .",
                   "Today ,"]
        expected_matrix = np.array([[1., 7./8., 0.],
                                    [7./9., 1., 2./9.]])
        expected_pred = candidates[1]
        result_pred, result_matrix = mbr(samples, unigram_precision, candidates=candidates, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # Test that it doesn't return a matrix by default.
        result_pred = mbr(samples, unigram_precision, candidates=candidates)
        self.assertEqual(result_pred, expected_pred)

        # Test with subsampling.
        samples = ["A beautiful red bike .",
                   "A precious yellow bike .",
                   "A beautiful yellow bike .",
                   "A precious red bike . "]
        result_pred, result_matrix = mbr(samples, unigram_precision, candidates=None, subsample_size=2, 
                                         return_matrix=True)
        self.assertEqual(result_matrix.shape, (4, 2))

        # Randomized test, should give different results depending on subsamples. Should pass with high probability.
        predictions = []
        for _ in range(100):
            pred = mbr(samples, unigram_precision, candidates=None, subsample_size=2)
            predictions += pred
        pred_vocab = set(predictions)
        self.assertTrue("yellow" in pred_vocab and "red" in pred_vocab)
