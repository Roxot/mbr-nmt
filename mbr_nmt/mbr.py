import numpy as np
import warnings

def unique_samples(samples):
    _, uniq_ids = np.unique(samples, return_index=True)
    return [samples[idx] for idx in uniq_ids]

def mbr(samples, utility, candidates=None, return_matrix=False, subsample_size=None, subsample_per_candidate=False,
        return_topk=False):
    """
    Maximizes the MBR objective for one sentence given a list of samples, utility function,
    and optionally a separate list of candidates.

    :param samples: a list of lists of strings, containing translation samples.
    :param utility: a utility function to maximize.
    :param candidates: optional, a list of lists of sentences representing translation candidates
                       to consider. If not given, assumed equal to samples.
    :param return_matrix: optional, boolean, if true this function additionally returns the utility matrix as a numpy.ndarray.
    :param subsample_size: optional, integer, if given a smaller uniformly sampled subsample is used to approximate
                           expectations for faster runtime.
    :param return_topk: optional, integer, returns top-k predictions rather than best only.
    """

    # If no candidates are given, we assume the samples to be the candidates.
    if candidates is None: candidates = unique_samples(samples)
    num_samples = len(samples)
    num_candidates = len(candidates)

    # Subsample a smaller amount of samples for each candidate if set.
    if subsample_size is not None:
        if subsample_size <= 0:
            raise Exception("Invalid subsample size.")
        if subsample_size > len(samples):
            warnings.warn("Subsample size is larger than the number of samples. Are you sure this is intended?")

        if not subsample_per_candidate:
            sample_indices = np.random.choice(len(samples), subsample_size, replace=True)
            subsample = [samples[idx] for idx in sample_indices]
        else:
            sample_indices = np.random.choice(num_samples, [num_candidates, subsample_size], replace=True)

        num_samples = subsample_size
    elif subsample_per_candidate:
        raise Exception("subsample_per_candidate can only be set if subsample_size > 0.")
    else:
        subsample = samples
        
    if utility.supports_batching:
        if utility.requires_tokenization:
            raise NotImplementedError()
        batched_candidates = np.repeat(candidates, num_samples, axis=0).tolist()
        if subsample_per_candidate:
            batched_samples = [samples[idx] for idx in sample_indices.flatten()]
        else:
            batched_samples = np.tile(subsample, num_candidates).tolist()
        utilities = utility.sentence_scores(batched_candidates, batched_samples)
        matrix = np.reshape(utilities, [num_candidates, num_samples])
    else:
        # Fill in the utility matrix.
        matrix = np.zeros([num_candidates, num_samples])

        # Do some pre-processing if necessary for more efficient utility assessments.
        if utility.requires_tokenization:
            tok_candidates = [utility.tokenizer(c) for c in candidates] 
            if subsample_size and subsample_per_candidate:
                tok_samples = [utility.tokenizer(s) for s in samples]
            else:
                tok_subsample = [utility.tokenizer(s) for s in subsample]
        else:
            tok_candidates = candidates
            tok_samples = samples
            tok_subsample = subsample

        for i, candidate in enumerate(tok_candidates):

            # use a different subsample per candidate
            if subsample_size and subsample_per_candidate:
                subsample = [tok_samples[idx] for idx in sample_indices[i]]

            for j, sample in enumerate(tok_subsample):
                matrix[i, j] = utility(hyp=candidate, ref=sample)

    # Compute E[utility(candidate, .)]
    expectation = np.average(matrix, axis=1)

    if return_topk:
        # Pick the top-k best translations according to expected utility.
        prediction_idx = np.argsort(expectation)[::-1][:return_topk]
        prediction = [candidates[pid] for pid in prediction_idx]
    else:
        # Pick the argmax as final translation.
        prediction_idx = np.argmax(expectation)
        prediction = candidates[prediction_idx]

    # TODO changed this, tests are broken now. (didn't return pred_idx before)
    if return_matrix:
        return prediction_idx, prediction, matrix
    else:
        return prediction_idx, prediction
