import unittest
import numpy as np 

from mbr_nmt.mbr import mbr
from mbr_nmt.utility import unigram_precision

class TestMBR(unittest.TestCase):

    def test_mbr(self):
        candidates = [["George went to school by bike today .".split(" "), 
                       "Today , George went to school by bike .".split(" "),  
                       "This is a completely unrelated sentence .".split(" ")],
                    ["It was a red bike .".split(" "),
                     "The bike was red .".split(" "),
                     "The bike .".split(" ")]]

        # Test matrix computations and prediction.
        expected_matrix = np.array([[1., 7./8., 1./8.],
                                    [7./9., 1., 1./9.],
                                    [1./7., 1./7., 1.]])
        expected_pred = candidates[0][0]
        result_pred, result_matrix = mbr(candidates[0], unigram_precision, samples=None, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # One more test on another set of candidates.
        expected_matrix = np.array([[1., 4./6., 2./6.],
                                    [4./5., 1., 3./5.],
                                    [2./3., 3./3., 1.]])
        expected_pred = candidates[1][2]
        result_pred, result_matrix = mbr(candidates[1], unigram_precision, samples=None, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # Test that separate candidates and samples work.
        candidates = ["It was a red bike .".split(" "),
                      "The bike was red .".split(" "),
                      "The bike .".split(" "),
                      "bike .".split()]
        samples = ["It was a red bike .".split(" "),
                   "The bike was red .".split(" "),
                   "The bike .".split(" ")]
        expected_matrix = np.array([[1., 4./6., 2./6.],
                                    [4./5., 1., 3./5.],
                                    [2./3., 3./3., 1.],
                                    [2./2., 2./2., 2./2.]])
        expected_pred = candidates[3]
        result_pred, result_matrix = mbr(candidates, unigram_precision, samples=samples, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # And when we have more samples than candidates.
        candidates = ["George went to school by bike today .".split(" "), 
                      "Today , George went to school by bike .".split(" ")]
        samples = ["George went to school by bike today .".split(" "), 
                   "Today , George went to school by bike .".split(" "),
                   "Today ,".split(" ")]
        expected_matrix = np.array([[1., 7./8., 0.],
                                    [7./9., 1., 2./9.]])
        expected_pred = candidates[1]
        result_pred, result_matrix = mbr(candidates, unigram_precision, samples=samples, return_matrix=True)
        self.assertEqual(result_pred, expected_pred)
        self.assertTrue((result_matrix == expected_matrix).all())

        # Test that it doesn't return a matrix by default.
        result_pred = mbr(candidates, unigram_precision, samples=samples)
        self.assertEqual(result_pred, expected_pred)
