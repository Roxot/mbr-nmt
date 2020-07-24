import argparse
import sys

from mbr_nmt.io import read_candidates_file
from mbr_nmt.utility import parse_utility
from mbr_nmt.mbr import mbr

def translate(args):
    fout = sys.stdout

    # Read and process input arguments.
    if args.candidates and not args.num_candidates:
        raise Exception("Must set --num-candidates if --candidates/-c is given.")
    S = read_candidates_file(args.samples, args.num_samples)
    C = read_candidates_file(args.candidates, args.num_candidates) if args.candidates else None
    if C is not None and len(C) != len(S):
        raise Exception("Different dataset size for candidates and samples.")
    utility = parse_utility(args.utility)

    # Run MBR on the entire dataset.
    for sequence_idx, samples in enumerate(S):
        candidates = C[sequence_idx] if C else None
        pred = mbr(samples, utility, 
                   candidates=candidates, 
                   return_matrix=False,
                   subsample_size=args.subsample_size)
        fout.write("{}\n".format(" ".join(pred)))


def create_parser(subparsers=None):
    if subparsers is None:
        parser = argparse.ArgumentParser()
    else:
        parser = subparsers.add_parser("translate")

    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="File containing translation samples, one per line, in order of input sequence.")
    parser.add_argument("--num-samples", "-n", type=int, required=True,
                        help="Number of samples per input sequence.")
    parser.add_argument("--utility", "-u", type=str, required=True,
                        help="Utility function to maximize.", choices=["unigram-precision", "beer"])
    parser.add_argument("--candidates", "-c", type=str,
                        help="File containing translation candidates, one per line, in order of input sequence. "
                             "If not given, assumed to be equal to --samples/-s.")
    parser.add_argument("--num-candidates", "-m", type=int,
                        help="Number of candidates per input sequence, only used if --candidates/-c is set.")
    parser.add_argument("--subsample-size", type=int,
                        help="If set, a smaller uniformly sampled subsample is used to compute expectations "
                             "for faster runtime.")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    translate(args)
