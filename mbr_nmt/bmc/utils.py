import numpy as np

class Sampler:
    """
    This is a helper class to abstract the NMT model as a generator.
    """
    
    def __init__(self, repository):
        self.repository = repository
        
    def __call__(self, num_samples, replace=True):
        """
        Draw num_samples from repository with replacement, 
        or min(len(repository), num_samples) without replacement. 
        
        When drawing without replacement, the order is always random.        
        """
        if replace:           
            ids = np.random.choice(len(self.repository), num_samples, replace=True)
        elif num_samples >= len(self.repository):
            ids = np.random.permutation(np.arange(len(self.repository)))
        else:
            ids = np.random.choice(len(self.repository), num_samples, replace=False)
        return [self.repository[i] for i in ids]
