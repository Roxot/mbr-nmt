import numpy as np
import warnings

def mbr(samples, utility, candidates=None, return_matrix=False, subsample_size=None):
    """
    Maximizes the MBR objective for one sentence given a list of samples, utility function,
    and optionally a separate list of candidates.

    :param samples: a list of lists of strings, containing tokenized translation samples.
    :param utility: a utility function to maximize.
    :param candidates: optional, a list of lists of sentences representing tokenized translation candidates
                       to consider. If not given, assumed equal to samples.
    :param return_matrix: optional, boolean, if true this function additionally returns the utility matrix as a numpy.ndarray.
    :param subsample_size: optional, integer, if given a smaller uniformly sampled subsample is used to approximate
                           expectations for faster runtime.
    """

    # If no candidates are given, we assume the samples to be the candidates.
    if candidates is None: candidates = samples
    num_samples = len(samples)

    # Subsample a smaller amount of samples for each candidate if set.
    if subsample_size is not None:
        if subsample_size <= 0:
            raise Exception("Invalid subsample size.")
        if subsample_size > len(samples):
            warnings.warn("Subsample size is larger than the number of samples. Are you sure this is intended?")
        sample_indices = np.random.randint(0, len(samples), size=[len(candidates), subsample_size])
        samples = np.array(samples)
        num_samples = subsample_size
        
    # Fill in the utility matrix.
    matrix = np.zeros([len(candidates), num_samples])
    for i, candidate in enumerate(candidates):
        if subsample_size is not None:
            subsample = samples[sample_indices[i]]
        else:
            subsample = samples
        
        for j, sample in enumerate(subsample):
            matrix[i, j] = utility(hyp=candidate, ref=sample)

    # Compute E[utility(candidate, .)]
    expectation = np.average(matrix, axis=1)

    # Pick the argmax as final translation.
    prediction_idx = np.argmax(expectation)
    prediction = candidates[prediction_idx]

    # TODO changed this, tests are broken now. (didn't return pred_idx before)
    if return_matrix:
        return prediction_idx, prediction, matrix
    else:
        return prediction_idx, prediction
