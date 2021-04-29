import subprocess
import os
import threading
import warnings
import sacrebleu
import bleurt
import tempfile
import re

import mbr_nmt

from itertools import combinations
from nltk.util import ngrams

from mbr_nmt.external.chrF import computeChrF

def parse_utility(string, lang=None, bleurt_checkpoint=None):
    if string == "unigram-precision":
        return NGramPrecision(1)
    elif string == "beer":
        return BEER()
    elif string == "meteor":
        if not METEOR.is_available_lang(lang):
            warnings.warn("Language {} unavailable for METEOR, "
                          "defaulting to {} instead.".format(lang, METEOR.default_lang))
            lang = METEOR.default_lang
        return METEOR(lang)
    elif string == 'bleu':
        return BLEU()
    elif string == "chrf":
        return ChrF()
    elif string == "chrf++":
        return ChrFPP()
    elif string == "ter":
        return TER()
    elif string == "bleurt":
        return BLEURT(bleurt_checkpoint)
    else:
        raise Exception("Unknown utility: " + string)

class NGramPrecision:

    def __init__(self, n):
        self.n = n

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(ngrams(hyp.split(' '), self.n))
        ref_set = set(ngrams(ref.split(' '), self.n))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(hyp_set) if hyp_set else 0.

class NGramRecall:

    def __init__(self, n):
        self.n = n

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(ngrams(hyp.split(' '), self.n))
        ref_set = set(ngrams(ref.split(' '), self.n))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(ref_set) if ref_set else 0.

class NGramF:

    def __init__(self, n):
        self.n = n

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(ngrams(hyp.split(' '), self.n))
        ref_set = set(ngrams(ref.split(' '), self.n))
        matches = hyp_set.intersection(ref_set)
        n = len(matches) 
        p = n / len(hyp_set) if len(hyp_set) else 0.
        r = n / len(ref_set) if len(ref_set) else 0.
        return 0. if (p + r) == 0. else 2. * p * r / (p + r)

class SkipBigramPrecision:
    
    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(combinations(hyp.split(' '), 2))
        ref_set = set(combinations(ref.split(' '), 2))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(hyp_set) if hyp_set else 0.0

    
class SkipBigramRecall:
    
    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(combinations(hyp.split(' '), 2))
        ref_set = set(combinations(ref.split(' '), 2))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(ref_set) if ref_set else 0.0   
    
class SkipBigramF:
    
    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(combinations(hyp.split(' '), 2))
        ref_set = set(combinations(ref.split(' '), 2))
        matches = hyp_set.intersection(ref_set)
        n = len(matches)
        p = n / len(hyp_set) if hyp_set else 0.0
        r = n / len(ref_set) if ref_set else 0.0
        return 0.0 if (p + r) == 0. else 2. * p * r / (p + r)


class BLEU:

    def __init__(self, smooth_method='floor', smooth_value=None, use_effective_order=True):
        self._smooth_method = smooth_method
        self._smooth_value = smooth_value
        self._use_effective_order = use_effective_order

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return sacrebleu.sentence_bleu(hyp, ref, smooth_method=self._smooth_method, 
                                       smooth_value=self._smooth_value, 
                                       use_effective_order=self._use_effective_order).score

class ChrF:

    def __init__(self, order=6, beta=2, remove_whitespace=True):
        self._order = order
        self._beta = beta
        self._remove_whitespace = remove_whitespace

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return sacrebleu.sentence_chrf(hyp, ref, order=self._order, 
                                       beta=self._beta, 
                                       remove_whitespace=self._remove_whitespace).score

class ChrFPP:

    def __init__(self, nworder=2, ncorder=6, beta=2.):
        self.nworder = nworder
        self.ncorder = ncorder
        self.beta = beta

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        if len(hyp) == 0 or len(ref) == 0: return 0.
        return computeChrF(fpRef=[ref], fpHyp=[hyp], nworder=self.nworder, ncorder=self.ncorder, beta=self.beta)[1]

class TER:

    def __init__(self, normalized=False, no_punct=False, asian_support=False, case_sensitive=False):
        self._normalized = normalized
        self._no_punct = no_punct
        self._asian_support = asian_support
        self._case_sensitive = case_sensitive

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        loss = sacrebleu.sentence_ter(hyp, [ref], normalized=self._normalized, 
               no_punct=self._no_punct, asian_support=self._asian_support,
               case_sensitive=self._case_sensitive).score
        return -loss

class BEER:

    def __init__(self, threads=4, model="default"):
        if "BEER_HOME" not in os.environ:
            raise Exception("For use of BEER as utility, make sure BEER is installed and "
                            "$BEER_HOME is set.")
        beer_home = os.environ["BEER_HOME"]
        self.proc = subprocess.Popen([beer_home + "/scripts/interactive", "-t", str(threads), "--model", model],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
        self.lock = threading.Lock()
        self.beer_home = beer_home
        self.threads = threads
        self.model = model

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        
        Returns BEER sentence score for hyp and ref.
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return self.sentence_score(hyp, ref)

    def sentence_score(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        if len(hyp) == 0 or len(ref) == 0: return 0.
        self.lock.acquire()
        self.proc.stdin.write("EVAL ||| {} ||| {}\n".format(hyp, ref).encode("utf-8"))
        self.proc.stdin.flush()
        beer = float(self.proc.stdout.readline())
        self.lock.release()
        return beer

    def corpus_score(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        """

        # Write hypotheses and references to a temp file.
        htmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        rtmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            for hyp in hyps: htmp.write(f"{hyp}\n")
            for ref in refs: rtmp.write(f"{ref}\n")
        finally:
            htmp.close()
            rtmp.close()

        # Compute corpus-METEOR score.        
        try:
            out = subprocess.run([self.beer_home + "/beer", "-t", str(self.threads),
                                  "-s", htmp.name, "-r", rtmp.name, "--model", self.model],
                                 stdout=subprocess.PIPE)
        finally:
            os.unlink(htmp.name)
            os.unlink(rtmp.name)

        return float(re.findall("\d+\.\d+", out.stdout.decode("utf-8"))[0])

    def __del__(self):
        """
        Make sure to close the subprocess when no longer needed.
        """
        if hasattr(self, "lock"):
            self.lock.acquire()
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
            self.lock.release()

class METEOR:

    default_lang = "en"
    fully_supported = ["en", "cz", "fr", "de", "es"]
    available_languages = ["en", "cz", "de", "es", "fr", "da", "fi", "hu", "it",
                           "nl", "no", "pt", "ro", "ru", "se", "tr", "other"]

    def __init__(self, lang, tokenize=None, custom_args=None):
        """
        :param lang: one of METEOR.available_languages, or other.
        :param tokenize: whether to use the built-in METEOR tokenizer. Only available for languages in METEOR.fully_supported.
        :param custom_args: additional custom args to be passed to METEOR.
        """
        if tokenize is None: tokenize = lang in METEOR.fully_supported
        meteor_folder = os.path.join(mbr_nmt.__path__[0], 'metrics/meteor')
        if not os.path.exists(meteor_folder):
            raise Exception("METEOR not installed, expect meteor-1.5.jar and data in {}".format(meteor_folder))
        if lang not in self.available_languages:
            raise Exception(f"lang parameter not one of available languages: {lang} not in {self.available_languages}")
        if tokenize and lang not in self.fully_supported:
            raise Exception(f"built-in tokenization only available for fully supported languages: {lang} not in {self.fully_supported}")       

        jar_file = os.path.join(meteor_folder, "meteor-1.5.jar")

        tok = ["-norm"] if tokenize else []
        custom = [custom_args] if custom_args else []
        self.proc = subprocess.Popen(["java", "-Xmx2G", "-jar", jar_file, "-", "-",
                                       "-stdio", "-l", lang] + tok + custom,
                                     cwd=os.path.dirname(os.path.abspath(__file__)),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.lock = threading.Lock()

        self.jar_file = jar_file
        self.lang = lang
        self.tokenize = tokenize
        self.custom_args = custom_args

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces.
        :param ref: string, single reference, tokens separated by spaces.  

        Returns METEOR sentence score for hyp and ref.
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return self.sentence_score(hyp, ref)

    def sentence_score(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces.
        :param ref: string, single reference, tokens separated by spaces.
        """
        
        self.lock.acquire()
        self.proc.stdin.write("SCORE ||| {} ||| {}\n".format(ref, hyp).encode("utf-8"))
        self.proc.stdin.flush()
        scores = self.proc.stdout.readline().decode("utf-8").rstrip()
        self.proc.stdin.write("EVAL ||| {}\n".format(scores).encode("utf-8"))
        self.proc.stdin.flush()
        meteor = float(self.proc.stdout.readline().strip())
        self.lock.release()
        return meteor

    def corpus_score(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        """

        # Write hypotheses and references to a temp file.
        htmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        rtmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            for hyp in hyps: htmp.write(f"{hyp}\n")
            for ref in refs: rtmp.write(f"{ref}\n")
        finally:
            htmp.close()
            rtmp.close()

        # Compute corpus-METEOR score.        
        try:
            tok = ["-norm"] if self.tokenize else []
            custom = [self.custom_args] if self.custom_args else []
            out = subprocess.run(["java", "-Xmx2G", "-jar", self.jar_file, htmp.name, rtmp.name,
                                         "-l", self.lang, "-q"] + tok + custom,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE)
        finally:
            os.unlink(htmp.name)
            os.unlink(rtmp.name)

        return float(out.stdout.strip())

    def __del__(self):
        if hasattr(self, "lock"):
            self.lock.acquire()
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.wait()
            self.lock.release()

    @staticmethod
    def is_available_lang(lang):
        return lang in METEOR.available_languages

class BLEURT:
    
    def __init__(self, checkpoint, batch_size=16):
        self.supports_batching = True
        self.batch_size = batch_size
        self.scorer = bleurt.score.BleurtScorer(checkpoint)

    def sentence_scores(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        
        Returns a list of sentence-level scores.
        """
        return scorer.score(refs, hyps, batch_size=batch_size) 
    
    def corpus_score(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        """
        return np.average(self.sentence_scores(hyps, refs))
