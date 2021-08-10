from mbr_nmt.bmc import BayesMCMBR, KernelSettings, Sampler
from mbr_nmt.mbr import unique_samples

import numpy as np 

def bayes_mc_mbr(samples, kernel_utility, utility, candidates=None, subsample_size=None, subsample_per_candidate=False,
                 return_gp_mean=False, return_topk=False):
    if candidates is None: candidates = unique_samples(samples)

    # Non-changeable default settings. (for now)
    kernel_settings = KernelSettings(kernel_utility)
    obs_var = 0.1
    mean_val = 0.5

    if utility.supports_batching: raise Warning("This utility is best used in batching mode. "
                                                "Batching mode is currently not supported for Bayesian MC MBR.")

    # Create the BMC MBR decoder.
    sampler = Sampler(samples)
    bmc_decoder = BayesMCMBR(kernel_settings, sampler, candidates, obs_var=obs_var,
                             mean=mean_val, lib="gpytorch", gpu=False)

    # Collect observations.
    if subsample_size is None:
        Y = np.array([[utility(h, r) for r in samples] for h in candidates])
    else:
        if subsample_per_candidate:
            Y = np.array([[utility(h, r) for r in sampler(subsample_size)] for h in candidates])
        else:
            subsample = sampler(subsample_size)
            Y = np.array([[utility(h, r) for r in subsample] for h in candidates])
    bmc_decoder.add_observations(observations=Y, indices=np.arange(len(candidates)))

    # Infer the GP posterior mean and translate.
    pred_idx, mbr_translation, gp_mean, gp_var = bmc_decoder.translate(return_topk=return_topk)

    if return_gp_mean:
        return pred_idx, mbr_translation, gp_mean
    else:
        return pred_idx, mbr_translation
