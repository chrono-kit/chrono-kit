import numpy as np
import torch
import scipy.optimize as opt
from scipy.stats import norm
from scipy.stats import linregress
from scipy.linalg import toeplitz
import warnings
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.preprocessing import differencing
from chronokit.preprocessing import AutoCorrelation
from chronokit.base._initializers import Initializer

class SARIMAInitializer(Initializer):

    def __init__(self, model):
        """
        This class uses; 
        * yule_walker equation for AR and seasonal AR parameters
        * least squares estimation on for MA and seasonal MA parameters

        This is wrong, however *usually* leads to reasonable parameters
        Can also lead to unstability on some cases of models
        """

        #TODO: Implement different initialization methods
        method = "default"
        
        super().__init__(model, method)

        self.used_params = {}
        self.init_params = {}
        
        if method != "default":
            self.err_func = {"mle": self._log_likelihood_err,
                             "simple": self._sse_err}[self.estimation_method]

    def __init_used_params(self):

        if self.model.d > 0:
            non_seasonal_data = differencing(self.model.data, order=self.model.d, return_numpy=False)
        else:
            non_seasonal_data = self.model.data.clone()
        
        if self.model.D > 0:
            seasonal_data = differencing(self.model.data[range(0, len(self.model.data), self.model.seasonal_periods)], 
                                         order=self.model.D, return_numpy=False)
        elif self.model.seasonal_periods is not None:
            seasonal_data = self.model.data[range(0, len(self.model.data), self.model.seasonal_periods)].clone()
        else:
            non_seasonal_data = self.model.data.clone()

        if self.model.p > 0:
            self.init_params["phi"] = self.__yule_walker(non_seasonal_data, order=self.model.p)
            self.used_params["phi"] = self.model.p
        else:
            self.init_params["phi"] = torch.zeros(0)

        if self.model.q > 0:
            self.init_params["theta"] = torch.zeros(self.model.q)
            self.used_params["theta"] = self.model.q
        else:
            self.init_params["theta"] = torch.zeros(0)
        
        if self.model.P > 0:
            self.init_params["seasonal_phi"] = self.__yule_walker(seasonal_data, order=self.model.P)
            self.used_params["seasonal_phi"] = self.model.P
        else:
            self.init_params["seasonal_phi"] = torch.zeros(0)
        
        if self.model.Q > 0:
            self.init_params["seasonal_theta"] = torch.zeros(self.model.Q)
            self.used_params["seasonal_theta"] = self.model.Q
        else:
            self.init_params["seasonal_theta"] = torch.zeros(0)
           
        for param, value in self.init_params.items():
                setattr(self.model, param, value)
        
    def __yule_walker(self, data, order):

        """
        Yule-Walker equations 
        for initializing parameters of an AR(p) model

        The equations were written by Chapter 7.3.1 of Box, G. et al.
        as reference
        """

        r = DataLoader(AutoCorrelation(data).acf(order)).to_tensor()
        # Get the toeplitz matrix of autocovariances
        # R = [1, r1, r2, ..., rp-2, rp-1]
        #    [r1, 1, r2, ..., rp-3, rp-2]
        #     ...       ....         ...
        #    [rp-2, rp-3, rp-4, .., 1, 0]
        #    [rp-1, rp-2, rp-3, .. r1, 1]
    
        R = torch.tensor(toeplitz(r[:-1]), dtype=torch.float32)

        # R is full-rank and symmetric therefore invertible
        # phi = R^(-1)r
        phi_params = torch.matmul(torch.linalg.inv(R), r[1:])

        return phi_params
    
    def __least_squares_on_theta(self):
        """Run least squares estimation on theta only"""
        def func(x):

            start_ind = 0
            for param in ["theta", "seasonal_theta"]:
                if param in self.used_params:
                    order = self.used_params[param]
                else:
                    continue
                values = x[start_ind:start_ind+order]
                setattr(self.model, param, values)
                start_ind += order
            
            self.model.fit()
            err = self.model.data - self.model.fitted

            return self._sse_err(err)
        
        init_values = [self.init_params[param] for param in ["theta", "seasonal_theta"]]
        init_values = DataLoader(torch.cat(init_values)).to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = opt.least_squares(fun=func, x0=init_values, max_nfev=10)

        result_params = list(results.x)
        self.success = results.success
        
        def assign_results(results):
            start_ind = 0
            for param in ["theta", "seasonal_theta"]:
                if param in self.used_params:
                    order = self.used_params[param]
                else:
                    continue
                values = results[start_ind:start_ind+order]
                self.init_params[param] = DataLoader(values).to_tensor()
                start_ind += order

        assign_results(result_params)


    def initialize_parameters(self):
        """Initialize model parameters"""
        self.__init_used_params()

        if self.estimation_method == "default":
            if self.model.q + self.model.Q > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.__least_squares_on_theta()
            else:
                self.success = True
            return self.init_params

        #Below is needed for future fixes on SARIMA initialization implementations
        #We will basically minimize sse or negative log likelihood on fit errors
        #For current starting estimates, this does not work as intended
        #So, only default initialization is available as of v1.1.x
        def func(x):

            start_ind = 0
            for param, order in self.used_params.items():
                values = x[start_ind:start_ind+order]
                setattr(self.model, param, values)
                start_ind += order
            
            self.model.fit()
            err = self.model.data - self.model.fitted

            return self.err_func(err)
        
        init_values = [self.init_params[param] for param in self.used_params.keys()]
        init_values = DataLoader(torch.cat(init_values)).to_numpy()

        if self.estimation_method == "simple":
            results = opt.least_squares(fun=func, x0=init_values)
        elif self.estimation_method == "mle":
            results = opt.minimize(fun=func, x0=init_values)
            
        result_params = list(results.x)
        self.success = results.success
        
        def assign_results(results):
            start_ind = 0
            for param, order in self.used_params.items():
                values = results[start_ind:start_ind+order]
                self.init_params[param] = DataLoader(values).to_tensor()
                start_ind += order

        assign_results(result_params)

        return self.init_params