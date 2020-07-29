import subprocess
import os
import threading
import warnings
import mbr_nmt

def parse_utility(string, lang=None):
    if string == "unigram-precision":
        return unigram_precision
    elif string == "beer":
        return BEER()
    elif string == "meteor":
        if not METEOR.is_available_lang(lang):
            warnings.warn("Language {} unavailable for METEOR, "
                          "defaulting to {} instead.".format(lang, METEOR.default_lang))
            lang = METEOR.default_lang
        return METEOR(lang)
    else:
        raise Exception("Unknown utility: " + string)

def unigram_precision(hyp, ref):
    """
    :param hyp: hypothesis, list of tokens (strings).
    :param ref: reference, list of tokens (strings).
    """
    hyp_set = set(hyp)
    matches = hyp_set.intersection(set(ref))
    return len(matches) / len(hyp_set)

class BEER:

    def __init__(self, threads=4):
        if "BEER_HOME" not in os.environ:
            raise Exception("For use of BEER as utility, make sure BEER is installed and "
                            "$BEER_HOME is set.")
        beer_home = os.environ["BEER_HOME"]
        self.proc = subprocess.Popen([beer_home + "/scripts/interactive", "-t", str(threads)],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
        self.lock = threading.Lock()

    def __call__(self, hyp, ref):
        """
        :param hyp: hypothesis, list of tokens (strings).
        :param ref: reference, list of tokens (strings).
        """
        self.lock.acquire()
        self.proc.stdin.write("EVAL ||| {} ||| {}\n".format(" ".join(hyp), " ".join(ref)).encode("utf-8"))
        self.proc.stdin.flush()
        beer = float(self.proc.stdout.readline())
        self.lock.release()
        return beer

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
    available_languages = ["en", "cz", "de", "es", "fr", "da", "fi", "hu", "it",
                           "nl", "no", "pt", "ro", "ru", "se", "tr"]

    def __init__(self, lang):
        meteor_folder = os.path.join(mbr_nmt.__path__[0], 'metrics/meteor')
        if not os.path.exists(meteor_folder):
            raise Exception("METEOR not installed, expect meteor-1.5.jar and data in {}".format(meteor_folder))
        jar_file = os.path.join(meteor_folder, "meteor-1.5.jar")
        self.proc = subprocess.Popen(["java", "-Xmx2G", "-jar", jar_file, "-", "-",
                                       "-stdio", "-l", lang],
                                     cwd=os.path.dirname(os.path.abspath(__file__)),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.lock = threading.Lock()

    def __call__(self, hyp, ref):
        self.lock.acquire()
        self.proc.stdin.write("SCORE ||| {} ||| {}\n".format(" ".join(ref), " ".join(hyp)).encode("utf-8"))
        self.proc.stdin.flush()
        scores = self.proc.stdout.readline().decode("utf-8").rstrip()
        self.proc.stdin.write("EVAL ||| {}\n".format(scores).encode("utf-8"))
        self.proc.stdin.flush()
        meteor = float(self.proc.stdout.readline().strip())
        self.lock.release()
        return meteor

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
