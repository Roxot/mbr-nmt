import subprocess
import os
import threading
import warnings
import sacrebleu
import tempfile
import re
import numpy as np
try:
    from bleurt import score as bleurt_score
except ImportError:
    bleurt_score = None 

try:
    import nepalitokenizer
except ImportError:
    nepalitokenizer = None 


import mbr_nmt

from itertools import combinations
from nltk.util import ngrams

from mbr_nmt.external.chrF import computeChrF

def parse_utility(string, lang=None, bleurt_checkpoint=None, tokenize=True):
    if string == "unigram-precision":
        return NGramPrecision(1, tokenize=tokenize)
    if string == "unigram-precision-symmetric":
        return NGramPrecisionSymmetricProd(1, tokenize=tokenize)
    if string == "bigram-precision":
        return NGramPrecision(2, tokenize=tokenize)
    elif string == "sum-1-to-4-ngram-precision-symmetric":
        return SumNGramPrecisionSymmetricProd(4, tokenize=tokenize)
    elif string == "unigram-f1":
        return NGramF(1, tokenize=(lang=="ne"))
    elif string == "bigram-f1":
        return NGramF(2, tokenize=(lang=="ne"))
    elif string == "sum-1-to-4-ngram-f1":
        return SumNGramF(4, tokenize=tokenize)
    elif string == "skip-bigram-precision":
        return SkipBigramPrecision(tokenize=tokenize)
    elif string == "skip-bigram-precision-symmetric":
        return SkipBigramPrecisionSymmetricProd(tokenize=tokenize)
    elif string == "skip-bigram-f1":
        return SkipBigramF(tokenize=False, lang=lang)
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
        if bleurt_score is None: raise Exception("BLEURT not installed.")
        if bleurt_checkpoint is None: raise Exception("Bleurt checkpoint not set.")
        return BLEURT(bleurt_checkpoint)
    else:
        raise Exception("Unknown utility: " + string)

class Utility:

    def __init__(self):

        # Whether this utility supports batching with self.sentence_scores(hyps, refs)
        self.supports_batching = False

        # Whether this utility requires tokenization as pre-processing step (requires self.tokenizer to be set).
        self.requires_tokenization = False
        self.tokenizer = None

    def sentence_scores(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        
        Returns a list of sentence-level scores. Required if self.supports_batching == True.
        """
        pass

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces

        returns the utility score of a single hypothesis, reference pair as float.
        """
        pass

class SumNGramPrecisionSymmetricProd(Utility):
 
    def __init__(self, n, tokenize=False):
        Utility.__init__(self)
        self.utilities = []
        for n_i in range(n):
            self.utilities.append(NGramPrecisionSymmetricProd(n, tokenize=tokenize))

    def __call__(self, hyp: str, ref: str):
        val = 0.
        for utility in self.utilities:
            val += utility(hyp, ref)
        return val
        
class SumNGramF(Utility):
 
    def __init__(self, n, tokenize=False):
        Utility.__init__(self)
        self.utilities = []
        for n_i in range(n):
            self.utilities.append(NGramF(n, tokenize=tokenize))

    def __call__(self, hyp: str, ref: str):
        val = 0.
        for utility in self.utilities:
            val += utility(hyp, ref)
        return val

class NGramPrecisionSymmetricProd(Utility):

    def __init__(self, n, tokenize=False):
        Utility.__init__(self)
        self.utility = NGramPrecision(n, tokenize=tokenize)

    def __call__(self, hyp: str, ref: str):
        return self.utility(hyp, ref) * self.utility(ref, hyp)

class NGramPrecision(Utility):

    def __init__(self, n, tokenize=False, lang="en"):
        Utility.__init__(self)
        self.n = n
        self.requires_tokenization = True

        if tokenize:
            if lang == "ne":
                # Nepali tokenizer special case
                if nepalitokenizer is None: raise Exception("nepalitokenizer not installed.")
                tok = nepalitokenizer.NepaliTokenizer()
                tokenize =  lambda s: ' '.join(tok.tokenizer(s))
            else:
                tokenize = sacrebleu.tokenize_13a
        else: tokenize = lambda x: x

        self.tokenizer = lambda s: set(ngrams(tokenize(s).split(' '), self.n))

    def __call__(self, hyp, ref):
        matches = hyp.intersection(ref)
        return len(matches) / len(hyp) if hyp else 0.

    def __str__(self):
        return f"{self.n}-GramPrecision"

class NGramRecall(Utility):

    def __init__(self, n, tokenize=False):
        Utility.__init__(self)
        self.n = n
        self.tokenize = tokenize

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        if self.tokenize:
            hyp = sacrebleu.tokenize_13a(hyp)
            ref = sacrebleu.tokenize_13a(ref)
        hyp_set = set(ngrams(hyp.split(' '), self.n))
        ref_set = set(ngrams(ref.split(' '), self.n))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(ref_set) if ref_set else 0.

class NGramF(Utility):

    def __init__(self, n, tokenize=False, lang="en"):
        Utility.__init__(self)
        self.n = n
        self.requires_tokenization = True

        if tokenize:
            if lang == "ne":
                # Nepali tokenizer special case
                if nepalitokenizer is None: raise Exception("nepalitokenizer not installed.")
                tok = nepalitokenizer.NepaliTokenizer()
                tokenize =  lambda s: ' '.join(tok.tokenizer(s))
            else:
                tokenize = sacrebleu.tokenize_13a
        else: tokenize = lambda x: x

        self.tokenizer = lambda s: set(ngrams(tokenize(s).split(' '), self.n))

    def __call__(self, hyp, ref):
        matches = hyp.intersection(ref)
        n = len(matches) 
        p = n / len(hyp) if len(hyp) else 0.
        r = n / len(ref) if len(ref) else 0.
        return 0. if (p + r) == 0. else 2. * p * r / (p + r)

class SkipBigramPrecision(Utility):

    def __init__(self, tokenize=False):
        Utility.__init__(self)
        self.tokenize = tokenize
    
    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        if self.tokenize:
            hyp = sacrebleu.tokenize_13a(hyp)
            ref = sacrebleu.tokenize_13a(ref)
        hyp_set = set(combinations(hyp.split(' '), 2))
        ref_set = set(combinations(ref.split(' '), 2))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(hyp_set) if hyp_set else 0.0

class SkipBigramPrecisionSymmetricProd(Utility):

    def __init__(self, tokenize=False):
        Utility.__init__(self)
        self.tokenize = tokenize
        self.utility = SkipBigramPrecision(tokenize=tokenize)
    
    def __call__(self, hyp: str, ref: str):
        return self.utility(hyp, ref) * self.utility(ref, hyp)

    
class SkipBigramRecall(Utility):

    def __init__(self, tokenize=False):
        Utility.__init__(self)
        self.tokenize = tokenize
    
    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        if self.tokenize:
            hyp = sacrebleu.tokenize_13a(hyp)
            ref = sacrebleu.tokenize_13a(ref)
        hyp_set = set(combinations(hyp.split(' '), 2))
        ref_set = set(combinations(ref.split(' '), 2))
        matches = hyp_set.intersection(ref_set)
        return len(matches) / len(ref_set) if ref_set else 0.0   
    
class SkipBigramF(Utility):

    def __init__(self, lang, tokenize=False):
        Utility.__init__(self)
        self.requires_tokenization = True

        if tokenize:
            if lang == "ne":
                # Nepali tokenizer special case
                if nepalitokenizer is None: raise Exception("nepalitokenizer not installed.")
                tok = nepalitokenizer.NepaliTokenizer()
                tokenize =  lambda s: ' '.join(tok.tokenizer(s))
            else:
                tokenize = sacrebleu.tokenize_13a
        else: tokenize = lambda x: x

        self.tokenizer = lambda s: set(combinations(tokenize(s).split(' '), 2))
    
    def __call__(self, hyp_set, ref_set):
        """
        :param hyp_set: pre-processed hyp
        :param ref_set: pre-processed ref
        """
        matches = hyp_set.intersection(ref_set)
        n = len(matches)
        p = n / len(hyp_set) if hyp_set else 0.0
        r = n / len(ref_set) if ref_set else 0.0
        return 0.0 if (p + r) == 0. else 2. * p * r / (p + r)


class BLEU(Utility):

    def __init__(self, smooth_method='floor', smooth_value=None, use_effective_order=True):
        Utility.__init__(self)
        self._smooth_method = smooth_method
        self._smooth_value = smooth_value
        self._use_effective_order = use_effective_order

    def corpus_score(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        """
        return sacrebleu.corpus_bleu(hyps, [refs]).score

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        return self.sentence_score(hyp, ref)

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return sacrebleu.sentence_bleu(hyp, ref, smooth_method=self._smooth_method, 
                                       smooth_value=self._smooth_value, 
                                       use_effective_order=self._use_effective_order).score

class ChrF(Utility):

    def __init__(self, order=6, beta=2, remove_whitespace=True):
        Utility.__init__(self)
        self._order = order
        self._beta = beta
        self._remove_whitespace = remove_whitespace


    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        return self.sentence_score(hyp, ref)

    def sentence_score(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        return sacrebleu.sentence_chrf(hyp, ref, order=self._order, 
                                       beta=self._beta, 
                                       remove_whitespace=self._remove_whitespace).score

    def corpus_score(self, hyps, refs):
        return sacrebleu.corpus_chrf(hyps, refs, order=self._order,
                                     beta=self._beta,
                                     remove_whitespace=self._remove_whitespace).score

class ChrFPP(Utility):

    def __init__(self, nworder=2, ncorder=6, beta=2.):
        Utility.__init__(self)
        self.nworder = nworder
        self.ncorder = ncorder
        self.beta = beta

    def corpus_score(self, hyps, refs):
        return computeChrF(fpRef=refs, fpHyp=hyps, nworder=self.nworder, ncorder=self.ncorder, beta=self.beta)[1]

    def __call__(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        return self.sentence_score(hyp, ref)

    def sentence_score(self, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        assert isinstance(hyp, str) and isinstance(ref, str)
        if len(hyp) == 0 or len(ref) == 0: return 0.
        try:
            chrf = computeChrF(fpRef=[ref], fpHyp=[hyp], nworder=self.nworder, ncorder=self.ncorder, beta=self.beta)[1]
        except Exception as e:
            chrf = 0.
        return chrf

class TER(Utility):

    def __init__(self, normalized=False, no_punct=False, asian_support=False, case_sensitive=False):
        Utility.__init__(self)
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

class BEER(Utility):

    def __init__(self, threads=4, model="default"):
        Utility.__init__(self)
        if "BEER_HOME" not in os.environ:
            raise Exception("For use of BEER as utility, make sure BEER is installed and "
                            "$BEER_HOME is set.")
        beer_home = os.environ["BEER_HOME"]
        self.proc = subprocess.Popen([beer_home + "/scripts/interactive", "-t", str(threads), "--model", model],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
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

        # Compute corpus-BEER score.        
        try:
            out = subprocess.run([self.beer_home + "/beer", "-t", str(self.threads),
                                  "-s", htmp.name, "-r", rtmp.name, "--model", self.model],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

class METEOR(Utility):

    default_lang = "other"
    fully_supported = ["en", "cz", "fr", "de", "es"]
    available_languages = ["en", "cz", "de", "es", "fr", "da", "fi", "hu", "it",
                           "nl", "no", "pt", "ro", "ru", "se", "tr", "other"]

    def __init__(self, lang, custom_args=None):
        """
        :param lang: one of METEOR.available_languages, or other.
        :param custom_args: additional custom args to be passed to METEOR.
        """
        Utility.__init__(self)
        meteor_folder = os.path.join(mbr_nmt.__path__[0], 'metrics/meteor')
        if not os.path.exists(meteor_folder):
            raise Exception("METEOR not installed, expect meteor-1.5.jar and data in {}".format(meteor_folder))
        if lang not in self.available_languages:
            raise Exception(f"lang parameter not one of available languages: {lang} not in {self.available_languages}")

        jar_file = os.path.join(meteor_folder, "meteor-1.5.jar")

        use_norm = lang in METEOR.fully_supported
        tok = ["-norm"] if use_norm else ["-lower"]

        custom = [custom_args] if custom_args else []
        self.proc = subprocess.Popen(["java", "-Xmx2G", "-jar", jar_file, "-", "-",
                                       "-stdio", "-l", lang] + tok + custom,
                                     cwd=os.path.dirname(os.path.abspath(__file__)),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.lock = threading.Lock()

        # If we can't use METEOR's built-in tokenizer, we need to tokenize ourselves.
        if not use_norm:
            if lang == "ne":
                # Nepali tokenizer special case
                if nepalitokenizer is None: raise Exception("nepalitokenizer not installed.")
                self.tokenizer = nepalitokenizer.NepaliTokenizer()
                self.tokenize =  lambda s: ' '.join(self.tokenizer.tokenizer(s))
            else:
                self.tokenize = sacrebleu.tokenize_13a

        self.jar_file = jar_file
        self.lang = lang
        self.use_norm = use_norm
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
        if not self.use_norm:
            hyp = self.tokenize(hyp)
            ref=  self.tokenize(ref)

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
            if self.use_norm:
                for hyp in hyps: htmp.write(f"{hyp}\n")
                for ref in refs: rtmp.write(f"{ref}\n")
            else:
                for hyp in hyps: htmp.write(f"{self.tokenize(hyp)}\n")
                for ref in refs: rtmp.write(f"{self.tokenize(ref)}\n")
        finally:
            htmp.close()
            rtmp.close()

        # Compute corpus-METEOR score.        
        try:
            tok = ["-norm"] if self.use_norm else ["-lower"]
            custom = [self.custom_args] if self.custom_args else []
            out = subprocess.run(["java", "-Xmx2G", "-jar", self.jar_file, htmp.name, rtmp.name,
                                         "-l", self.lang, "-q"] + tok + custom,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
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

class BLEURT(Utility):
    
    def __init__(self, checkpoint, batch_size=16):
        Utility.__init__(self)
        self.supports_batching = True
        self.batch_size = batch_size
        self.scorer = bleurt_score.BleurtScorer(checkpoint)

    def sentence_scores(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        
        Returns a list of sentence-level scores.
        """
        return self.scorer.score(references=refs, candidates=hyps, batch_size=self.batch_size) 
    
    def corpus_score(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.
        """
        return np.average(self.sentence_scores(hyps, refs))
