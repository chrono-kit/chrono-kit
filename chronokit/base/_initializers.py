import numpy as np
import pandas as pd
import torch
import scipy.optimize as opt
import warnings
from scipy.stats import norm
from scipy.stats import linregress
from scipy.linalg import toeplitz
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.preprocessing.data_transforms import differencing
from chronokit.preprocessing.autocorrelations import AutoCorrelation

class Initializer:

    def __init__(self, model, method):
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