import numpy as np

def compute_utility_features(inputs, pivots, utility, template='r'):
    """
    :param inputs: list of list of sentences representing tokens for each sentence
    :param pivots: list of list of sentences representing tokens for each sentence, each pivot is an unbiased sample
    :param utility function used to construct the kernel
    :param template:
        'r' introduces features of the kind utility(hyp=input, ref=pivot)
        'h' introduces features of the kind utility(hyp=pivot, ref=input)
        'rh' introduces both types
    """    

    features = []
    # work around to avoid empty strings (which seem to break some utilities)

    # Note: depending on the utility we might be able to optimise this (e.g., unigram precision)
    
    # Compute features \phi: 
    #  \phi_r = utility(hyp=x, ref=pivot) 
    #  \phi_h = utility(hyp=pivot, ref=x)
    if template == 'r':
        for x in inputs:
            features.append(np.array([[utility(x, pivot)] for pivot in pivots]))                
    elif template == 'h':
        for x in inputs:
            features.append(np.array([[utility(pivot, x)] for pivot in pivots]))                
    elif template == 'rh':
        for x in inputs:
            features.append(np.array([[utility(x, pivot), utility(pivot, x)] for pivot in pivots]))
    else:
        raise Exception("Unknown template, use 'r', 'h', or 'rh'.")
    features = np.array(features)   
    return features
