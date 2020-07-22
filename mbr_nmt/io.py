from mbr_nmt.tokenizer import default_tokenizer

def read_candidates_file(filename, num_candidates, tokenizer=default_tokenizer):
    candidates = []
    with open(filename, "r") as f:
        for line_id, line in enumerate(f.readlines()):
            if line_id % num_candidates == 0:
                if line_id != 0: candidates.append(candidates_i)
                candidates_i = []
            candidates_i.append(tokenizer(line.rstrip()))
        candidates.append(candidates_i)

    if len(candidates[-1]) != num_candidates:
        raise Exception("Invalid candidate file, did you specify the number of candidates correctly?")

    return candidates
