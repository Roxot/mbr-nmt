from mbr_nmt.bmc.utils import Sampler
from mbr_nmt.bmc.features import compute_utility_features
from mbr_nmt.bmc.gp import SparseGP
import numpy as np

class KernelSettings:
    
    def __init__(self, 
                 utility,
                 ktype='rbf',
                 features='r',
                 combinator=None,
                 npivots=500,
                 unit_multiplier=None,
                 rbf_variance=0.1,
                 rbf_lengthscale=5.):
        self.utility = utility
        self.features = features
        self.combinator = combinator
        self.npivots = npivots
        self.unit_multiplier = unit_multiplier
        self.ktype = ktype
        self.rbf_variance = rbf_variance
        self.rbf_lengthscale = rbf_lengthscale

class BayesMCMBR:
    
    def __init__(self, 
                 kernel_settings: KernelSettings,
                 sampler: Sampler,
                 hyps,
                 obs_var = 0.1,
                 mean = 0.5,
                 optimize=False,
                 lib="gpytorch",
                 gpu=False
                 ):

        # Draw pivots (each is a dimension of the string kernel)
        pivots = sampler(kernel_settings.npivots)
        
        # [num_hypotheses, num_kernels * num_pivots] 
        X = compute_utility_features(
                inputs=hyps, 
                pivots=pivots, 
                utility=kernel_settings.utility, 
                template=kernel_settings.features).reshape([len(hyps), -1])

        # [num_hypotheses * mc_samples, num_kernels * num_pivots]
        X_rep = X

        # construct the GP.
        gp = SparseGP(X, kernel_settings, mean, obs_var, lib=lib, gpu=gpu)
        
        self.gp = gp
        self.kernel_dim = kernel_settings.npivots*len(kernel_settings.features)
        self.kernel_settings = kernel_settings
        self.mean = mean
        self.obs_var = obs_var
        self.X = X
        self.X_rep = X_rep
        self.hyps = np.array(hyps)
        self.Y_rep = None
        self.optimize = optimize
        self.lib = lib
        self.gpu = gpu

    def prior(self, include_obs_var):
        return self.gp.prior(include_obs_var)
        
    def prior_samples(self, size, include_obs_var):
        mean, cov = self.prior(include_obs_var)
        return np.random.multivariate_normal(mean, cov, size=size)
    
    def add_observations(self, observations, indices):
        """
        - observations: observations of shape [len(indices), number of observations]
        - indices: indices in original list of hypotheses (hyp_space) to get additional observations for, list of integers
        - num: number of additional observations to collect, integer
        """
        H = self.hyps[indices] 
        num_observations = observations.shape[1]
        
        # [len(indices) *  num, 1]
        Y_rep_prime = observations.reshape([-1, 1])
        
        # Add to existing observations
        if self.Y_rep is None:
            self.Y_rep = Y_rep_prime
            
            if num_observations > 1:
                
                # [len(indices), num_kernels * num_pivots] 
                X_prime = self.X[indices, :]

                # [len(indices) * num, num_kernels * num_pivots]
                X_rep_prime = X_prime.repeat(num_observations, axis=0)
                self.X_rep = X_rep_prime
            
        else:
            self.Y_rep = np.concatenate([self.Y_rep, Y_rep_prime], axis=0)
            
            # [len(indices), num_kernels * num_pivots] 
            X_prime = self.X[indices, :]

            # [len(indices) * num, num_kernels * num_pivots]
            X_rep_prime = X_prime.repeat(num_observations, axis=0)
            
            self.X_rep = np.concatenate([self.X_rep, X_rep_prime], axis=0)

        # Update the GP
        self.gp.infer(self.X_rep, self.Y_rep, optimize=self.optimize)

    def posterior(self, include_obs_var, full_cov=False):
        return self.gp.posterior(include_obs_var, full_cov=full_cov)

    def posterior_samples(self, size, include_obs_var):
        mean, cov = self.posterior(include_obs_var, full_cov=True)
        return np.random.multivariate_normal(mean.flatten(), cov, size=size)
    
    def translate(self, return_mean_var=False, return_topk=False):
        mean, var = self.posterior(include_obs_var=False, full_cov=False)  

        if return_topk:
            prediction_idx = np.argsort(mean)[::-1][:return_topk]
            prediction = [self.hyps[pid] for pid in prediction_idx]
        else:
            prediction_idx = np.argmax(mean)
            prediction = self.hyps[prediction_idx]
            
        return prediction_idx, prediction, mean, var
