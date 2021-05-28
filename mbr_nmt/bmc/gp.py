try:
    import GPy
except ImportError:
    GPy = None

try:
    import gpytorch
except ImportError:
    gpytorch = None

import torch
import numpy as np

class SparseGP:

    def __init__(self, Z, kernel_settings, mean, obs_var, lib="gpytorch", gpu=False):
        """
        :param Z: inducing points design matrix
        :param kernel_settings: KernelSettings object
        :param mean: constant mean value
        :param obs_var: observation variance
        :param lib: whether to use GPy or gpytorch
        :param gpu: whether to use GPU acceleration (only gpytorch)
        """
        lib = lib.lower()
        if gpu and lib != "gpytorch": raise Exception("GPU acceleration only available for lib=gpytorch.")
        if lib == "gpy":
            if GPy is None: raise Exception("GPy not installed.")
            self.kernel = construct_kernel_gpy(kernel_settings)
            self.Z = Z
        elif lib == "gpytorch":
            if gpytorch is None: raise Exception("gpytorch not installed.")
            self.kernel = construct_kernel_gpytorch(kernel_settings)   
            self.Z = torch.from_numpy(Z.astype(np.float32))

            # Add some Gaussian noise to the features for numerical stability.
            self.Z = self.Z + 1e-6 * torch.randn(self.Z.shape)

            if gpu:
                self.Z = self.Z.cuda()
                self.kernel.cuda()   
        else:
            raise NotImplementedError

        self.mean = mean
        self.obs_var = obs_var
        self.lib = lib
        self.gpu = gpu

    def infer(self, X, Y, optimize=False):
        """
        :param X: design matrix
        :param Y: observations
        """
        if self.lib == "gpy":

            # Construct the mean function
            mean_fn = GPy.core.Mapping(self.kernel.input_dim, 1)
            mean_fn.f = lambda x: self.mean
            mean_fn.update_gradients = lambda a, b: 0.
            mean_fn.gradients_X = lambda a, b: 0.

            # Set up the GP
            gp = GPy.models.SparseGPRegression(X, Y, Z=self.Z ,
                                kernel=self.kernel,
                                normalizer=False, 
                                mean_function=mean_fn)
            gp.likelihood.variance = self.obs_var
            
            # Optimize marginal likelihood
            if optimize:
                gp.optimize()

            self.gp = gp
        elif self.lib == "gpytorch":

            X = torch.from_numpy(X.astype(np.float32))
            Y = torch.from_numpy(Y.astype(np.float32)).reshape([-1])

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            with torch.no_grad():
                likelihood.noise = torch.Tensor([self.obs_var])

            # Add some Gaussian noise to the features for numerical stability.
            X = X + 1e-6 * torch.randn(X.shape)

            if self.gpu: 
                X = X.cuda()
                Y = Y.cuda()

            self.gp = GPyTorchGP(X, Y, self.Z, likelihood, self.kernel)
            with torch.no_grad():
                self.gp.mean_module.constant = torch.nn.Parameter(torch.Tensor([self.mean]))
            if self.gpu:
                self.gp = self.gp.cuda()

            if optimize:
                self.gp.train()
                likelihood.train()

                # Use the adam optimizer
                optimizer = torch.optim.Adam([{"params": self.gp.base_covar_module.parameters()}], lr=0.01)

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp)

                training_iter = 1000
                for i in range(training_iter):
                    optimizer.zero_grad()
                    output = self.gp(X)
                    loss = -mll(output, Y)
                    loss.backward()
                    optimizer.step()    

                    if i % 500 == 0 or i == training_iter-1:
                        print('Iter %d/%d - Loss: %.5f mean: %.3f  var: %.3f lengthscale: %.3f   noise: %.3f' % (
                            i + 1, training_iter, loss.item(),
                            self.gp.mean_module.constant.item(),
                            self.gp.base_covar_module.outputscale.item(),
                            self.gp.base_covar_module.base_kernel.lengthscale.item(),
                            self.gp.likelihood.noise.item()
                        ))

            self.gp.eval()
            likelihood.eval()
        else:
            raise NotImplementedError

    def prior(self, include_obs_var):
        if self.lib == "gpy":
            mean = np.full(self.Z.shape[0], self.mean)
            cov = self.kernel.K(self.Z)
            if include_obs_var:
                cov += (np.eye(cov.shape[0]) * self.obs_var)
            return mean, cov
        elif self.lib == "gpytorch":
            with torch.no_grad():
                mean = np.full(self.Z.shape[0], self.mean)
                cov = self.kernel(self.Z).cpu().numpy()
                if include_obs_var:
                    cov += (np.eye(cov.shape[0]) * self.obs_var)
            return mean, cov
        else:
            raise NotImplementedError

    def posterior(self, include_obs_var, full_cov=True):
        if self.gp is None:
            raise Exception("infer(X, Y) should be called first.")

        if self.lib == "gpy":
            if include_obs_var:
                mean, var = self.gp.predict(self.Z, full_cov=full_cov)
            else:
                mean, var = self.gp.predict_noiseless(self.Z, full_cov=full_cov)
            
            mean = mean.flatten()
            var = var.flatten() if not full_cov else var

            return mean, var
        elif self.lib == "gpytorch":
            with torch.no_grad():#, gpytorch.settings.max_preconditioner_size(10):
            #with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
                posterior = self.gp(self.Z)

                if include_obs_var:
                    posterior = self.gp.likelihood(posterior)

                mean = posterior.mean.cpu().numpy().flatten()
                    
                if full_cov:
                    var = posterior.covariance_matrix.cpu().numpy()
                else:
                    var = posterior.variance.cpu().numpy().flatten()

            return mean, var
        else:
            raise NotImplementedError

    def posterior_marginals(self, include_obs_var):
        return self.posterior(include_obs_var, full_cov=False)

class GPyTorchGP(gpytorch.models.ExactGP):

    def __init__(self, X, Y, Z, likelihood, kernel):
        super(GPyTorchGP, self).__init__(X, Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = kernel
        self.covar_module = self.base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def construct_kernel_gpytorch(kernel_settings):
    ktype = kernel_settings.ktype.lower()
    if ktype == 'linear':
        raise NotImplementedError
    elif ktype == 'rbf':
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        with torch.no_grad():
           kernel.outputscale = torch.Tensor([kernel_settings.rbf_variance]) 
           kernel.base_kernel.lengthscale = torch.Tensor([kernel_settings.rbf_lengthscale])
    else:
        raise NotImplementedError
    return kernel

def construct_kernel_gpy(kernel_settings):
    kernel = None
    num_kernels = len(kernel_settings.features) # one or two directions of kernel utilities
    ktype = kernel_settings.ktype.lower()
    if ktype == 'linear':
        if 'r' in kernel_settings.features:
            if kernel_settings.combinator == 'add': 
                variance = kernel_settings.unit_multiplier / num_kernels
            else: 
                variance = kernel_settings.unit_multiplier
            variance /= kernel_settings.npivots
            k_r = GPy.kern.Linear(
                input_dim=kernel_settings.npivots, 
                variances=variance,
                active_dims=np.arange(0, kernel_settings.npivots))
            kernel = k_r
        if 'h' in kernel_settings.features:
            if kernel_settings.combinator == 'add': 
                variance = kernel_settings.unit_multiplier / num_kernels
            else:
                variance = 1.
            variance /= kernel_settings.npivots
            k_h = GPy.kern.Linear(
                input_dim=kernel_settings.npivots, 
                variances=variance,
                active_dims=np.arange(kernel_settings.npivots * (num_kernels - 1), 
                                      kernel_settings.npivots * num_kernels)
            )
            if kernel is None:
                kernel = k_h
            elif kernel_settings.combinator == 'prod':
                kernel *= k_h
            else:
                kernel += k_h
    elif ktype == 'rbf':
        raise NotImplementedError
        # kernel = GPy.kern.RBF(input_dim=kernel_settings.npivots*num_kernels,
        #                       variance=kernel_settings.rbf_variance, 
        #                      lengthscale=kernel_settings.rbf_lengthscale)
    else:
        raise NotImplementedError
    return kernel
