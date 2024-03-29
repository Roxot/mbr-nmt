import argparse
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from sacremoses import MosesDetruecaser
from sacremoses import MosesTokenizer, MosesDetokenizer

from mbr_nmt.io import wc
from mbr_nmt.processing import merge_subwords as merge

def convert(args):
    if args.input_format == "fairseq":
        convert_from_fairseq(args.input_files, args.output_file, args.output_format,
                             merge_subwords=args.merge_subwords,
                             detruecase=args.detruecase,
                             detokenize=args.detokenize,
                             lang=args.lang,
                             encoding=args.encoding)
    elif args.input_format == "mbr-nmt":
        if len(args.input_files) > 1:
            raise exception("Multiple input files not supported for mbr-nmt format.")
        convert_mbr_translations(args.input_files[0], args.output_file, merge_subwords=args.merge_subwords,
                                 detokenize=args.detokenize, detruecase=args.detruecase, lang=args.lang, encoding=args.encoding)
    else:
        raise Exception("Unknown input format: {}".format(args.input_format))

def convert_mbr_translations(input_file, output_file, merge_subwords=False, detruecase=False,
                             detokenize=False, lang="en", encoding=None):
    sent_ids = []
    translations = []
    
    if detruecase:
        detruecaser = MosesDetruecaser()
    if detokenize:
        detokenizer = MosesDetokenizer(lang)

    with open(input_file, "r", encoding=encoding) as fi:
        for line in fi:
            try:
                sent_idx, translation, pred_idx = line.split(" ||| ")
                sent_ids.append(int(sent_idx))
                translation = translation.strip()
                if merge_subwords:
                    translation = merge(translation, style="fairseq")
                if detruecase:
                    translation = detruecaser.detruecase(translation, return_str=True)
                if detokenize:
                    translation = detokenizer.detokenize(translation.split(' '))
                translations.append(translation)
            except:
                raise Exception("Invalid file format, expects lines like 'sentence_id ||| translation'")

    sort_ids = np.argsort(sent_ids)
    with open(output_file, "w", encoding=encoding) as fo:
        for sort_idx in sort_ids:
            sent_idx = sent_ids[sort_idx]
            fo.write(f"{translations[sort_idx]}\n")

def convert_from_fairseq(input_files, output_file, output_format, merge_subwords=False, 
                         detruecase=False, detokenize=False, lang="en",
                         verbose=True, encoding=None):
    if output_format not in ["samples", "candidates"]:
        raise Exception(f"Invalid output format {output_format} for input format fairseq.")
    hyps = defaultdict(list)
    num_lines = sum(wc(input_file) for input_file in input_files)

    if detruecase:
        detruecaser = MosesDetruecaser()
    if detokenize:
        detokenizer = MosesDetokenizer(lang)
    
    # Parse the fairseq input files.
    if verbose: print("parsing input files...")
    if verbose: pbar = tqdm(total=num_lines)
    try:
        for input_file in input_files:
            with open(input_file, "r", encoding=encoding) as fi:
                for line in fi:
                    if verbose: pbar.update(1)
                    if line[:2] != "H-": continue
                    line = line.rstrip().split()
                    idx = int(line[0].split("-")[1])
                    likelihood = line[1]
                    tokens = line[2:]
                    hyp = ' '.join(tokens)
                    if merge_subwords:
                        hyp = merge(hyp, style="fairseq")
                    if detruecase:
                        hyp = detruecaser.detruecase(hyp, return_str=True)
                    if detokenize:
                        hyp = detokenizer.detokenize(hyp.split(' '))
                    hyps[idx].append(hyp)
    finally:
        if verbose: pbar.close()

    # Write in the output format format to the output file.
    if isinstance(output_file, str):
        fo = open(output_file, "w", encoding=encoding)
    else:
        fo = output_file
    if verbose: print("writing output to {}...".format(output_file))

    if output_format == "samples":
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
    else: # candidates
        avg_candidates = 0
        Z = len(hyps)
        try:
            for idx in sorted(hyps.keys()):
                num_hyps = len(hyps[idx])
                fo.write(f"NC={num_hyps}\n")
                for hyp in hyps[idx]:
                    fo.write("{}\n".format(hyp))
                avg_candidates += num_hyps / Z
        finally:
            if isinstance(output_file, str): fo.close()
        if verbose: print("found an average of {:.1f} candidates per input sequence".format(avg_candidates))

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
    parser.add_argument("--output-file", "-o", type=str, required=True,
                        help="The destination output file of hypotheses stored in mbr-nmt format.")
    parser.add_argument("--input-format", "-f", type=str, default="fairseq",
                        help="Input file format.", choices=["fairseq", "mbr-nmt"])
    parser.add_argument("--output-format", type=str, default="translations",
                        help="Output file format: samples, candidates or translations.", choices=["samples", "candidates", "translations"])
    parser.add_argument("--merge-subwords", action="store_true",
                        help="Merges subwords in the translations.")
    parser.add_argument("--detruecase", action="store_true",
                        help="Detruecase translations using the Moses detruecaser.")
    parser.add_argument("--detokenize", action="store_true",
                        help="Detokenize translations using the Moses detokenizer.")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language used for the Moses detokenizer.")
    parser.add_argument("--encoding", type=str,
                        help="File encoding for io operations.")

    parser.set_defaults(merge_subwords=False, detruecase=False, detokenize=False)

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    convert(args)
