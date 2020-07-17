def read_candidates_file(filename, num_candidates):
    candidates = []
    with open(filename, "r") as f:
        for line_id, line in enumerate(f.readlines()):
            if line_id % num_candidates == 0:
                if line_id != 0: candidates.append(candidates_i)
                candidates_i = []
            candidates_i.append(line.rstrip())
        candidates.append(candidates_i)

    if len(candidates[-1]) != num_candidates:
        raise Exception("Invalid candidate file, did you specify the number of candidates correctly?")

    return candidates
