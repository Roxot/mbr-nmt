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
from mbr_nmt.c2f import c2f_mbr
from mbr_nmt.bayesmc import bayes_mc_mbr

def translate(args):
    finfo = sys.stderr
    finfo.write(f"{str(args)}\n")

    if args.bmc and args.kernel_utility is None: raise Exception("Kernel utility not set.")

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
            sent_idx, translation, pred_idx, expected_utility = queue.get()
            if sent_idx < 0: break
            fout.write("{} ||| {} ||| {}\n".format(sent_idx, translation, pred_idx))
            if expected_utility_folder:
                np.save(expected_utility_folder / f"exp-util-{sent_idx}.npy", expected_utility)
    finally:
        if filename: fout.close()

def run_mbr(S, C, start_idx, args, writer_queue):
    utility = parse_utility(args.utility, lang=args.lang, bleurt_checkpoint=args.bleurt_checkpoint)
    if args.c2f:
        if args.utility_2:
            utility2 = parse_utility(args.utility_2, lang=args.lang, bleurt_checkpoint=args.bleurt_checkpoint)
        else:
            utility2 = utility

    for sequence_idx, samples in enumerate(S):
        candidates = C[sequence_idx] if C else None

        if args.bmc:
            kernel_utility = parse_utility(args.kernel_utility, lang=args.lang, 
                                           bleurt_checkpoint=args.bleurt_checkpoint)
        else:
            kernel_utility = None

        if args.c2f:
            pred_idx, pred, utility_matrix = c2f_mbr(samples,
                                                     utility1=utility,
                                                     topk=args.top_k,
                                                     utility2=utility2,
                                                     candidates=candidates, 
                                                     mc1=args.subsample_size, mc2=args.subsample_size_2,
                                                     return_matrix=True,
                                                     subsample_per_candidate=args.subsample_per_candidate,
                                                     bmc=args.bmc, kernel_utility=kernel_utility)
            exp_utility = utility_matrix.mean(axis=1)
        elif args.bmc:
            pred_idx, pred, exp_utility = bayes_mc_mbr(samples, kernel_utility, utility, 
                                                       candidates=candidates,
                                                       subsample_size=args.subsample_size,
                                                       subsample_per_candidate=args.subsample_per_candidate,
                                                       return_gp_mean=True)
        else:
            pred_idx, pred, utility_matrix = mbr(samples, utility, 
                                                 candidates=candidates, 
                                                 return_matrix=True,
                                                 subsample_size=args.subsample_size,
                                                 subsample_per_candidate=args.subsample_per_candidate)
            exp_utility = utility_matrix.mean(axis=1)
        writer_queue.put((start_idx+sequence_idx, pred, pred_idx, exp_utility))

def create_parser(subparsers=None):
    available_utilities=["beer", "meteor",
                         "bleu", "chrf", "chrf++", "bleurt",
                         "unigram-precision", "unigram-f1", 
                         "unigram-precision-symmetric",
                         "skip-bigram-precision", "skip-bigram-f1",
                         "skip-bigram-precision-symmetric",
                         "sum-1-to-4-ngram-precision-symmetric",
                         "sum-1-to-4-ngram-f1"]
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
                        help="Utility function to maximize.", choices=available_utilities)
    parser.add_argument("--candidates", "-c", type=str,
                        help="File containing translation candidates, one per line preceded by the number of "
                             "candidates (e.g. NC=300), in order of input sequence. "
                             "If not given, assumed to be equal to --samples/-s.")
    parser.add_argument("--lang", "-l", type=str, default="en",
                        help="Language code used to inform METEOR.")
    parser.add_argument("--subsample-size", "-mc1", type=int,
                        help="If set, a smaller uniformly sampled subsample is used to compute expectations "
                             "for faster runtime (or in the coarse step of coarse-to-fine MBR if --c2f is set).")
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
    parser.add_argument("--bleurt-checkpoint", type=str, default=None, required=False,
                        help="The BLEURT checkpoint to use.")

    # Coarse-to-fine MBR
    parser.add_argument("--c2f", action="store_true",
                        help="Run MBR in two rounds.")
    parser.add_argument("--top-k", type=int,
                        help="Keep only the top-k candidates from the coarse step of coarse-to-fine MBR as candidates "
                             "in the fine step of coarse-to-fine MBR.")
    parser.add_argument("--subsample-size-2", "-mc2", type=int, required=False,
                        help="If set, a smaller uniformly sampled subsample is used to comput expectations "
                             "in the fine step of coarse-to-fine MBR.")
    parser.add_argument("--utility-2", "-u2", type=str, required=False, choices=available_utilities,
                        help="Utility function to maximize in the fine step of coarse-to-fine MBR. "
                             "If not set, will use --utility/-u instead.")

    # Bayesian MC
    parser.add_argument("--bmc", action="store_true",
                        help="Use Bayesian MC to estimate expected utility. In a coarse-to-fine setting this"
                             "will only be used in the coarse step.")
    parser.add_argument("--kernel-utility", default=None,
                        help="Utility used to compute features for Bayesian MC.")

    parser.set_defaults(subsample_per_candidate=False, c2f=False, bmc=False)
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    translate(args)
