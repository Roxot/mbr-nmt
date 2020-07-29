import argparse
from collections import defaultdict
from tqdm import tqdm

from mbr_nmt.io import wc

def convert(args):
    if args.input_format == "fairseq":
        convert_from_fairseq(args.input_files, args.output_file)
    else:
        raise Exception("Unknown input format: {}".format(args.input_format))

def convert_from_fairseq(input_files, output_file, verbose=True):
    hyps = defaultdict(list)
    num_lines = sum(wc(input_file) for input_file in input_files)
    
    # Parse the fairseq input files.
    if verbose: print("parsing input files...")
    if verbose: pbar = tqdm(total=num_lines)
    try:
        for input_file in input_files:
            with open(input_file, "r") as fi:
                for line in fi:
                    if verbose: pbar.update(1)
                    if line[:2] != "H-": continue
                    line = line.rstrip().split()
                    idx = int(line[0].split("-")[1])
                    likelihood = line[1]
                    tokens = line[2:]
                    hyps[idx].append(" ".join(tokens))
    finally:
        if verbose: pbar.close()

    # Write in mbr-nmt format to the output file.
    if isinstance(output_file, str):
        fo = open(output_file, "w")
    else:
        fo = output_file
    if verbose: print("writing output to {}...".format(output_file))
    num_hyps = None
    try:
        for idx in sorted(hyps.keys()):
            if num_hyps is None:
                num_hyps = len(hyps[idx])
            elif num_hyps != len(hyps[idx]):
                raise Exception("Unequal number of hypotheses per input sequence.")

            for hyp in hyps[idx]:
                fo.write("{}\n".format(hyp))
    finally:
        if isinstance(output_file, str): fo.close()
    if verbose: print("found an equal {} hypotheses per input sequence".format(num_hyps))

def create_parser(subparsers=None):
    description = "mbr-nmt convert: converts input files to the desired input format required by `mbr-nmt translate`"
    if subparsers is None:
        parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        parser = subparsers.add_parser("convert", description=description,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input-files", "-i", type=str, nargs="+", required=True,
                        help="A list of input files separated by spaces.")
    parser.add_argument("--output_file", "-o", type=str, required=True,
                        help="The destination output file of hypotheses stored in mbr-nmt format.")
    parser.add_argument("--input-format", "-f", type=str, default="fairseq",
                        help="Input file format.", choices=["fairseq"])

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    convert(args)
