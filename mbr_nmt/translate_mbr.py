import argparse

from mbr_nmt.io import read_candidates_file

def main(args):
    candidates = read_candidates_file(args.candidates, args.num)
    samples = read_candidates_file(args.samples, args.num) if args.samples else candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", "-c", type=str, required=True,
                        help="File containing a list of translation candidates.")
    parser.add_argument("--num", "-n", type=int, required=True,
                        help="Number of candidates per input sequence.")
    parser.add_argument("--samples", "-s", type=str, 
                        help="File containing a list of translation samples.")
    args = parser.parse_args()
    main(args)
