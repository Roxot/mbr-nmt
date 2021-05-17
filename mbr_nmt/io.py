import re

EOS_TOKEN = "</s>"

def read_samples_file(filename, num_samples, add_eos=False):
    samples = []
    with open(filename, "r") as f:
        for line_id, line in enumerate(f.readlines()):
            if line_id % num_samples == 0:
                if line_id != 0: samples.append(samples_i)
                samples_i = []
            line = line.rstrip()
            if add_eos: line = f"{line} {EOS_TOKEN}"
            samples_i.append(line)
        samples.append(samples_i)

    if len(samples[-1]) != num_samples:
        raise Exception("Invalid candidate file, did you specify the number of samples correctly?")

    return samples

def read_candidates_file(filename, add_eos=False):
    candidates = []
    candidates_i = None
    with open(filename, "r") as f:
        for line_id, line in enumerate(f.readlines()):
            if candidates_i is None:
                try:
                    ncandidates_i = int(line.rstrip()[3:])
                    candidates_i = []
                    continue
                except:
                    raise Exception("Invalid candidate file, expected candidate count NC=X on first line.")

            if re.match("^NC=[0-9]+", line):
                if ncandidates_i != len(candidates_i):
                    raise Exception(f"Invalid candidate file, expected {ncandidates_i} "
                                    f"candidates, found {len(candidates_i)}.")
                candidates.append(candidates_i)
                candidates_i = []
                ncandidates_i = int(line.rstrip()[3:])
            else:
                line = line.rstrip()
                if add_eos: line = f"{line} {EOS_TOKEN}"
                candidates_i.append(line)

        if ncandidates_i != len(candidates_i):
            raise Exception(f"Invalid candidate file, expected {ncandidates_i} "
                            f"candidates, found {len(candidates_i)}.")
        candidates.append(candidates_i)

    return candidates

def wc(filename):
    count = 0
    with open(filename, "r") as f:
        count += sum(1 for line in f)
    return count
