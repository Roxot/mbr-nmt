import unittest

from mbr_nmt.io import read_samples_file, EOS_TOKEN, wc

class TestIO(unittest.TestCase):

    def test_read_candidates(self):
        # Test that things are properly tokenized.
        samples = read_samples_file("test/data/samples-4.en", 4)
        expected = [["George went to school by bike today .".split(" "), 
                     "Today , George went to school by bike .".split(" "),
                     "This is a completely unrelated sentence .".split(" "),
                     "George went to school .".split(" ")],
                    ["He got the bike for his birthday .".split(" "),
                     "He got the bike for his anniversary .".split(" "),
                     "The birthday bike .".split(" "),
                     "bike .".split(" ")]]
        self.assertEqual(samples, expected)

        # Test that it raises an exception if a wrong number of samples is provided.
        raised_exception = False
        try:
            samples = read_samples_file("test/data/samples-4.en", 5)
        except:
            raised_exception = True
        self.assertTrue(raised_exception)

        # Test with a file containing empty sequences.
        samples = read_samples_file("test/data/samples-empty-3.en", 3)
        expected = [["George went to school by bike today .".split(" "),
                    [],
                    "This is a completely unrelated sentence .".split(" ")],
                    ["He got the bike for his birthday .".split(" "),
                     "The birthday bike .".split(" "),
                    []]]
        self.assertEqual(samples, expected)

        # Test add-eos.
        samples = read_samples_file("test/data/samples-empty-3.en", 3, add_eos=True)
        expected = [["George went to school by bike today .".split(" ") + [EOS_TOKEN],
                    [EOS_TOKEN],
                    "This is a completely unrelated sentence .".split(" ") + [EOS_TOKEN]],
                    ["He got the bike for his birthday .".split(" ") + [EOS_TOKEN],
                     "The birthday bike .".split(" ") + [EOS_TOKEN],
                    [EOS_TOKEN]]]
        self.assertEqual(samples, expected)

    def test_wc(self):
        result = wc("test/data/5lines.txt")
        expected = 5
        self.assertEqual(result, expected)
