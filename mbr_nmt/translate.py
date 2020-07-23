import argparse
import sys

from mbr_nmt.io import read_candidates_file
from mbr_nmt.utility import parse_utility
from mbr_nmt.mbr import mbr

def translate(args):
    fout = sys.stdout

    # Read and process input arguments.
    if args.samples and not args.num_samples:
        raise Exception("Must set --num-samples if --samples/-s is given.")
    candidates = read_candidates_file(args.candidates, args.num_candidates)
    samples = read_candidates_file(args.samples, args.num_samples) if args.samples else None
    if samples is not None and len(candidates) != len(samples):
        raise Exception("Different dataset size for candidates and samples.")
    utility = parse_utility(args.utility)

    # Run MBR on the entire dataset.
    for sequence_idx, candidate_set in enumerate(candidates):
        sample_set = samples[sequence_idx] if samples else None
        pred = mbr(candidate_set, utility, 
                   samples=sample_set, 
                   return_matrix=False,
                   subsample_candidates=args.subsample_candidates)
        fout.write("{}\n".format(" ".join(pred)))


def create_parser(subparsers=None):
    if subparsers is None:
        parser = argparse.ArgumentParser()
    else:
        parser = subparsers.add_parser("translate")

    parser.add_argument("--candidates", "-c", type=str, required=True,
                        help="File containing a list of translation candidates.")
    parser.add_argument("--num-candidates", "-n", type=int, required=True,
                        help="Number of candidates per input sequence.")
    parser.add_argument("--utility", "-u", type=str, required=True,
                        help="Utility function to maximize.", choices=["unigram-precision", "beer"])
    parser.add_argument("--samples", "-s", type=str, 
                        help="File containing a list of translation samples. "
                             "Assumed to be equal to candidates if not given.")
    parser.add_argument("--num-samples", "-m", type=int,
                        help="Number of samples per input sequence, only used if --samples/-s is set.")
    parser.add_argument("--subsample-candidates", type=int,
                        help="If set, will subsample the given amount of candidates to estimate expecatations.")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    translate(args)
