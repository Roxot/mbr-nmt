import argparse
import sys
import time
import numpy as np
import multiprocessing
import random

from pathlib import Path

from mbr_nmt.io import read_samples_file, read_candidates_file
from mbr_nmt.utility import parse_utility
from mbr_nmt.mbr import mbr

def translate(args):
    finfo = sys.stderr

    # Read and process input arguments.
    S = read_samples_file(args.samples, args.num_samples, add_eos=args.add_eos)
    C = read_candidates_file(args.candidates, add_eos=args.add_eos) if args.candidates else None
    if C is not None and len(C) != len(S):
        raise Exception("Different dataset size for candidates and samples.")

    if args.store_expected_utility:
        exp_utility_folder = Path(args.store_expected_utility)
        exp_utility_folder.mkdir(exist_ok=True)
    else:
        exp_utility_folder = None

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Run MBR on the entire dataset.
    start_time = time.time()

    threads = args.threads
    if args.threads <= 0:
        threads = min(multiprocessing.cpu_count(), len(S))
    if args.threads > len(S):
        raise Exception("Using more threads than translation candidates.")

    writer_queue = multiprocessing.Queue()
    writer_process = multiprocessing.Process(target=writer_job, args=(writer_queue, args.output_file, exp_utility_folder))
    writer_process.start()

    # Split up the input data into multiple threads.
    num_per_thread = int(np.ceil(len(S) / threads))
    idx = 0
    processes = []
    finfo.write(f"Starting {threads} parallel processes.\n")
    for t in range(threads):
        S_t = S[idx:idx+num_per_thread]
        if C: C_t = C[idx:idx+num_per_thread]
        else: C_t = None
        process = multiprocessing.Process(target=run_mbr, args=(S_t, C_t, idx, args, writer_queue))
        processes.append(process)
        process.start()
        idx += num_per_thread
        finfo.write(f"Started process {t+1} for doing {len(S_t)} translations.\n")

    # Wait for all processes to end.
    for process in processes:
        process.join()
    writer_queue.put((-1, "", -1, None))
    writer_process.join()

    finfo.write(f"Decoding took {time.time() - start_time:.0f}s\n")

def writer_job(queue, filename, expected_utility_folder):
    if filename: fout = open(filename, "w")
    else: fout = sys.stdout
    
    try:
        while True:
            sent_idx, translation, pred_idx, utility_matrix = queue.get()
            if sent_idx < 0: break
            fout.write("{} ||| {} ||| {}\n".format(sent_idx, translation, pred_idx))
            if expected_utility_folder:
                expected_utility = utility_matrix.mean(axis=1)
                np.save(expected_utility_folder / f"exp-util-{sent_idx}.npy", expected_utility)
    finally:
        if filename: fout.close()

def run_mbr(S, C, start_idx, args, writer_queue):
    utility = parse_utility(args.utility, lang=args.lang)

    for sequence_idx, samples in enumerate(S):
        candidates = C[sequence_idx] if C else None
        pred_idx, pred, utility_matrix = mbr(samples, utility, 
                                             candidates=candidates, 
                                             return_matrix=True,
                                             subsample_size=args.subsample_size,
                                             subsample_per_candidate=args.subsample_per_candidate)
        writer_queue.put((start_idx+sequence_idx, pred, pred_idx, utility_matrix))

def create_parser(subparsers=None):
    description = "mbr-nmt translate: pick an optimal translation according to minimum Bayes risk decoding"
    if subparsers is None:
        parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        parser = subparsers.add_parser("translate", description=description,
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--samples", "-s", type=str, required=True,
                        help="File containing translation samples, one per line, in order of input sequence.")
    parser.add_argument("--num-samples", "-n", type=int, required=True,
                        help="Number of samples per input sequence.")
    parser.add_argument("--utility", "-u", type=str, required=True,
                        help="Utility function to maximize.", choices=["unigram-precision", "beer", "meteor",
                                                                       "bleu", "chrf", "chrf++"])
    parser.add_argument("--candidates", "-c", type=str,
                        help="File containing translation candidates, one per line preceded by the number of "
                             "candidates (e.g. NC=300), in order of input sequence. "
                             "If not given, assumed to be equal to --samples/-s.")
    parser.add_argument("--lang", "-l", type=str, default="en",
                        help="Language code used to inform METEOR.")
    parser.add_argument("--subsample-size", type=int,
                        help="If set, a smaller uniformly sampled subsample is used to compute expectations "
                             "for faster runtime.")
    parser.add_argument("--add-eos", action="store_true",
                        help="Add an EOS token to every sample and candidate. "
                             "This is useful for dealing with empty sequences.")
    parser.add_argument("--threads", "-t", type=int, required=False, default=-1,
                        help="The number of threads to run. Depending on your CPU this can significantly "
                              "speed up computation. By default uses all available CPUs.")
    parser.add_argument("--output-file", "-o", type=str, required=False, default=None,
                        help="File to output translations to.")
    parser.add_argument("--store-expected-utility", type=str, required=False, default=None,
                       help="Folder to optionally store expected utility vectors.")
    parser.add_argument("--subsample-per-candidate", action="store_true",
                        help="Use a different subsample for each candidate.")
    parser.add_argument("--seed", type=int, default=None, required=False,
                        help="An optional random seed.")
    parser.set_defaults(subsample_per_candidate=False)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    translate(args)
