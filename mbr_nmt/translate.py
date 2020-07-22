import argparse

from mbr_nmt.io import read_candidates_file

def translate(args):
    candidates = read_candidates_file(args.candidates, args.num)
    samples = read_candidates_file(args.samples, args.num) if args.samples else candidates

def create_parser(subparsers=None):
    if subparsers is None:
        parser = argparse.ArgumentParser()
    else:
        parser = subparsers.add_parser("translate")

    parser.add_argument("--candidates", "-c", type=str, required=True,
                        help="File containing a list of translation candidates.")
    parser.add_argument("--num", "-n", type=int, required=True,
                        help="Number of candidates per input sequence.")
    parser.add_argument("--samples", "-s", type=str, 
                        help="File containing a list of translation samples.")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    translate(args)
