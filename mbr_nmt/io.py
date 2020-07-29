from mbr_nmt.tokenizer import default_tokenizer

EOS_TOKEN = "</s>"

def read_samples_file(filename, num_samples, tokenizer=default_tokenizer, add_eos=False):
    samples = []
    eos_token = [EOS_TOKEN] if add_eos else []
    with open(filename, "r") as f:
        for line_id, line in enumerate(f.readlines()):
            if line_id % num_samples == 0:
                if line_id != 0: samples.append(samples_i)
                samples_i = []
            tokenized_line = tokenizer(line.rstrip())
            if tokenized_line == [""]: tokenized_line = []
            samples_i.append(tokenized_line + eos_token)
        samples.append(samples_i)

    if len(samples[-1]) != num_samples:
        raise Exception("Invalid candidate file, did you specify the number of samples correctly?")

    return samples

def wc(filename):
    count = 0
    with open(filename, "r") as f:
        count += sum(1 for line in f)
    return count
