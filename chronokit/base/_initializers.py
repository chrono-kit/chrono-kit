import numpy as np
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader

class Initializer:

    def __init__(self, model, method):
        """
        Base Initializer class for initializing
        time series models

        All specialized initialization classes
        should inherit from this class
        """
        
        self.model = model
        self.estimation_method = method
        
        self.used_params = {}
        self.init_params = {}
    
    def _log_likelihood_err(self, err):
        err = DataLoader(err).to_numpy()
        err = err[~np.isnan(err)]

        loc = np.nanmean(err)
        scale = np.nanstd(err)
        log_likelihood = np.sum(norm.logpdf(err, loc=loc, scale=scale))
        
        return -log_likelihood

    def _sse_err(self, err):
        err = DataLoader(err).to_numpy()
        err = err[~np.isnan(err)]

        return np.sum(np.square(err))

    def initialize_parameters(self):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")