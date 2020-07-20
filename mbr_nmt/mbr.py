import numpy as np

def mbr(candidates, utility, samples=None, return_matrix=False):
    """
    Maximizes the MBR objective for one sentence given a list of candidates, utility function,
    and optionally a separate list of samples.

    :param candidates: a list of sentences representing translation candidates.
    :param utility: a utility function to maximize.
    :param samples: optional, a list of sentences containing sampled translations used to approximate the expectation. If not given, assumed equal to candidates.
    """
    if samples is None: samples = candidates

    # Compute the utility matrix.
    matrix = np.zeros([len(candidates), len(samples)])
    for i, candidate in enumerate(candidates):
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
