def unigram_precision(hyp, ref):
    """
    :param hyp: hypothesis, list of tokens (strings).
    :param ref: reference, list of tokens (strings).
    """
    hyp_set = set(hyp)
    matches = hyp_set.intersection(set(ref))
    return len(matches) / len(hyp_set)
