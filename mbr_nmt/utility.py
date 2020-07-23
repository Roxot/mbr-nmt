import subprocess
import os

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

    def __call__(self, hyp, ref):
        """
        :param hyp: hypothesis, list of tokens (strings).
        :param ref: reference, list of tokens (strings).
        """
        self.proc.stdin.write(("EVAL ||| " + " ".join(hyp) + " ||| " + \
                               " ".join(ref) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        return float(self.proc.stdout.readline())

    def close(self):
        """
        Make sure to close the subprocess by calling this function when finished.
        """
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.terminate()
        self.proc.wait()
