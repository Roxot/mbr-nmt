import unittest
import tempfile

from mbr_nmt.convert import convert_from_fairseq

class TestConvert(unittest.TestCase):

    def test_convert_from_fairseq(self):
        tmp = tempfile.TemporaryFile(mode="w+")
        try:
            convert_from_fairseq(["test/data/fairseq/samples-2_1.en", "test/data/fairseq/samples-2_2.en"], 
                                 tmp, verbose=False)
            tmp.seek(0)
            result = tmp.readlines()
            with open("test/data/samples-4.en", "r") as f: expected = f.readlines()
            self.assertEqual(result, expected)
        finally: tmp.close()
