import numpy as np

def promising_indices(means, variances, nstd):
    stds = np.sqrt(variances)
    intervals_upper = means + nstd * stds
    best_idx = np.argmax(means)
    best_lower = means[best_idx] - nstd * stds[best_idx]
    promising = np.where(intervals_upper >= best_lower)[0]
    return promising

class UniformAcquisition:
    
    def __init__(self, N, only_promising=True, num_std=3):
        """
        - N: number of samples per call to the acquistion function.
        - only_promising: if true only considers hypotheses whose posterior intervals overlap with
                          the best hypothesis
        - num_std: number of standard deviations that define the posterior interval
                   used if only_promising=True.
        """
        self.N = N
        self.only_promising = only_promising
        self.num_std = num_std
        self.cov = False
            
    def __call__(self, means, variances):
        """
        - means: GP posterior means of the latent mean. [num_hyps]
        - variances: GP marginal posterior variances of the latent mean.  
                     Recommended to not include observation variance. [num_hyps]
        
        Returns:
        - np.array() of indices (may contain repeated indices) of hypotheses to next collect observations from.
        """
        assert len(means.shape) == 1
        assert len(variances.shape) == 1
        if self.only_promising:
            promising = promising_indices(means, variances, self.num_std)
            return np.random.choice(promising, size=self.N, replace=True)
        else:
            return np.random.choice(len(means), size=self.N, replace=True)

    def __str__(self):
        if self.only_promising:
            return f"Uniform Acquisition - promising only (N={self.N})"
        else:
            return f"Uniform Acquisition (N={self.N})"

class ThompsonAcquisition:

    def __init__(self, N):
        """
        - N: number of samples per call to the acquistion function.
        """
        self.N = N
        self.cov = True
              
    def __call__(self, mean, cov):
        """
        - mean: GP posterior mean. [num_hyps]
        - cov: GP covariance matrix of the latent mean.  
        
        Returns:
        - np.array() of indices (may contain repeated indices) of hypotheses to next collect observations from.
        """
        assert len(mean.shape) == 1
        assert len(cov.shape) == 2

        # [N, nhyps]
        samples = np.random.multivariate_normal(mean, cov, size=self.N)
        
        # [nhyps]
        return np.argmax(samples, axis=-1)

    def __str__(self):
        return f"Thompson Acquisition (N={self.N})"

class TopKAcquisition:
    
    def __init__(self, N, k):
        """
        - N: number of samples per call to the acquistion function.
        - k: top-k hypotheses to consider according to GP mean.
        """
        self.N = N
        self.k = k
        self.cov = False
        
    def __call__(self, means, _):
        """
        - mean: GP posterior mean. [num_hyps]
        
        Returns:
        - np.array() of indices (may contain repeated indices) of hypotheses to next collect observations from.
        """
        topk = np.argsort(means)[::-1][:self.k]
        return np.random.choice(topk, size=self.N, replace=True)
    
    
    def __str__(self):
        return f"Top-{self.k} Acquisition (N={self.N})"
