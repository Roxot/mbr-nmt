import numpy as np

def mbr(candidates, utility, samples=None, return_matrix=False, subsample_candidates=None):
    """
    Maximizes the MBR objective for one sentence given a list of candidates, utility function,
    and optionally a separate list of samples.

    :param candidates: a list of sentences representing translation candidates.
    :param utility: a utility function to maximize.
    :param samples: optional, a list of sentences containing sampled translations used to approximate the expectation. If not given, assumed equal to candidates.
    :param return_matrix: optional, boolean, if true this function additionally returns the utility matrix as a numpy.ndarray.
    :param subsample_candidates: optional, integer, if given `samples` will be subsampled with replacement from `candidates`.
    """

    # Decide on how we will obtain MC samples to estimate E[utility(candidate, .)].
    if subsample_candidates is not None:
        if samples is not None: raise Exception("Cannot subsample candidates and use a fixed set of samples.")
        if subsample_candidates > len(candidates): raise Warning("Subsample size is larger than the number of candidates. Are you sure this is intended?")
        sample_indices = np.random.randint(0, len(candidates), size=[len(candidates), subsample_candidates])
        candidates = np.array(candidates)
        num_samples = subsample_candidates
    elif samples is None:
        samples = candidates
        num_samples = len(candidates)
    else:
        num_samples = len(samples)

    # Fill in the utility matrix.
    matrix = np.zeros([len(candidates), num_samples])
    for i, candidate in enumerate(candidates):
        if subsample_candidates is not None:
            samples = candidates[sample_indices[i]]
        
        for j, sample in enumerate(samples):
            matrix[i, j] = utility(hyp=candidate, ref=sample)

    # Compute E[utility(candidate, .)]
    expectation = np.average(matrix, axis=1)

    # Pick the argmax as final translation.
    prediction = candidates[np.argmax(expectation)]

    if return_matrix:
        return prediction, matrix
    else:
        return prediction
