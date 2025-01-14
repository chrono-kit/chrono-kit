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

    def __init__(self, model, method):
        """
        This class uses; 
        * yule_walker equation for AR and seasonal AR parameters
        * least squares estimation on for MA and seasonal MA parameters

        This is wrong, however *usually* leads to reasonable parameters
        Can also lead to unstability on some cases of models
        """
        
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
        
        if self.model.seasonal_periods is not None:
            if self.model.D > 0:
                non_seasonal_data = non_seasonal_data[self.model.seasonal_periods:] - non_seasonal_data[:-self.model.seasonal_periods]
      
            seasonal_data = non_seasonal_data[range(0, len(non_seasonal_data), self.model.seasonal_periods)]

        if self.model.p > 0 and self.model.q == 0: #AR
            self.init_params["phi"] = self.__yule_walker(non_seasonal_data, order=self.model.p)
            self.used_params["phi"] = self.model.p

            self.init_params["theta"] = torch.zeros(0)
     
        elif self.model.q > 0 and self.model.p == 0: #MA
            #r = DataLoader(AutoCorrelation(non_seasonal_data).acf(self.model.q)).to_tensor()
            self.init_params["theta"] = self.__initial_ma_estimates(non_seasonal_data, order=self.model.q)
            self.used_params["theta"] = self.model.q

            self.init_params["phi"] = torch.zeros(0)
       
        elif self.model.p > 0 and self.model.q > 0: #ARMA
            params = self.__initial_arma_estimates(non_seasonal_data, 
                                                   (self.model.p, self.model.q))

            self.init_params["phi"] = params[:-self.model.q]
            self.used_params["phi"] = self.model.p

            self.init_params["theta"] = params[-self.model.q:]
            self.used_params["theta"] = self.model.q
        
        else:
            self.init_params["phi"] = torch.zeros(0)
            self.init_params["theta"] = torch.zeros(0)

        
        if self.model.P > 0 and self.model.Q == 0:
            self.init_params["seasonal_phi"] = self.__yule_walker(seasonal_data, order=self.model.P, flag="seasonal ")
            self.used_params["seasonal_phi"] = self.model.P
        
            self.init_params["seasonal_theta"] = torch.zeros(0)
        
        elif self.model.Q > 0 and self.model.P == 0:
            self.init_params["seasonal_theta"] = self.__initial_ma_estimates(seasonal_data, order=self.model.Q, flag="seasonal ")
            self.used_params["seasonal_theta"] = self.model.q

            self.init_params["seasonal_phi"] = torch.zeros(0)
        
        elif self.model.P > 0 and self.model.Q > 0: #ARMA
            params = self.__initial_arma_estimates(seasonal_data, 
                                                   (self.model.P, self.model.Q), flag="seasonal ")

            self.init_params["seasonal_phi"] = params[:-self.model.Q]
            self.used_params["seasonal_phi"] = self.model.P

            self.init_params["seasonal_theta"] = params[-self.model.Q:]
            self.used_params["seasonal_theta"] = self.model.Q
        
        else:
            self.init_params["seasonal_phi"] = torch.zeros(0)
            self.init_params["seasonal_theta"] = torch.zeros(0)
           
        for param, value in self.init_params.items():
                setattr(self.model, param, value)
        
    def __yule_walker(self, data, order, flag=""):

        """
        Yule-Walker equations 
        for initializing parameters of an AR(p) model

        The equations were written by Chapter 7.3.1 of Box, G. et al.
        as reference
        """

        #First index is 1 (lag0)
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

        if not self.model.stationarity_condition(phi_params.numpy()):
            warnings.warn(f"Non Stationary initial {flag}AR estimates, \
                            initial estimates will be set to 0.\nHowever, \
                            it is recommended to choose a different order")
            phi_params = torch.zeros(order)

        return phi_params
    
    def __initial_ma_estimates(self, data, order, flag=""):
        """
        Estimates MA parameters by substituting autocorrelations of the data
        to the theoretical autocorrelation function.

        Equation for the theoretical autocorrelations were written by equation
        6.3.1 of Box, G. et al. as reference.
        """
        
        #0th index is 1
        r = DataLoader(AutoCorrelation(data).acf(order)).to_numpy()[1:]

        def func(x):

            numerator = np.empty(len(r))

            for i in range(1, len(r)+1):
                numerator[i-1] = -x[i-1]  + np.sum(x[:len(r)-i]*x[i:])

            numerator = numerator/(1 + np.sum(x**2))

            return numerator - r
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = opt.fsolve(func, x0=np.zeros(order))
        
        if not self.model.invertibility_condition(sol):
            warnings.warn(f"Non Invertible initial {flag}MA estimates, \
                            initial estimates will be set to 0.\nHowever, \
                            it is recommended to choose a different order")
            sol = np.zeros(order)

        return DataLoader(sol).to_tensor()
    
    def __initial_arma_estimates(self, data, order:tuple, flag=""):
        """
        Estimates ARMA parameters by substituting autocovariances of tthe data
        to the theoretical autocovariance function.

        Equations for the theoretical autocovariances were written by equations
        3.3.8 and 3.3.9 as reference from:

        Brockwell, Peter J., and Richard A. Davis. 2009. Time Series:
        Theory and Methods. 2nd ed. 1991. New York, NY: Springer.
        """
        p, q = order
        rt = np.empty(p+q+3)
        gamma0 = np.cov(data, data)[0, 1]

        rt[0] = gamma0
        for i in range(1,p+q+3):
            rt[i] = np.cov(data[i:], data[:-i])[0, 1]

        rts = np.concatenate((rt[-1::-1][:-1], rt))
        sigma = np.nanvar(data)

        def func(x):
            theta = np.concatenate((np.array([-1.]), x[-q:]))
            phi = x[:-q]

            psi_vals = np.ones(len(theta))
                
            for i in range(1, len(psi_vals)):
                phivals = phi.copy()
                if len(phivals) > len(psi_vals[:i]):
                    phivals = phivals[:i]
                elif len(phivals) < len(psi_vals[:i]):
                    phivals = np.concatenate((phivals, np.zeros(i-len(phivals))))

                psi_vals[i] = np.sum(phivals[-1::-1]*psi_vals[:i])

                if i <= len(theta):
                    psi_vals[i] = psi_vals[i] - theta[i]
            
            results_arr = np.zeros(len(rt))
         
            for k in range(len(results_arr)):
                if k < q + 1:
                    rhs = sigma*np.sum(theta[k:]*psi_vals[:q+1-k])
                else:
                    rhs = 0
                lhs = np.sum(rts[k+len(rt)-1-len(phi):k+len(rt)-1][-1::-1]*phi)
                
                results_arr[k] = rt[k] - lhs + rhs

            return np.sum(np.square(results_arr))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = opt.least_squares(func, x0=np.zeros(p+q)).x
    
        try:
            if not self.model.stationarity_condition(sol[:-q]):
                warnings.warn(f"Non Stationary initial {flag}AR estimates, \
                              initial estimates will be set to 0.\nHowever, \
                              it is recommended to choose a different order")
                sol[:-q] = np.zeros(p)
        except:
            sol[:-q] = np.zeros(p)
        
        try:
            if not self.model.invertibility_condition(sol[-q:]):
                warnings.warn(f"Non Invertible initial {flag}MA estimates, \
                              initial estimates will be set to 0.\nHowever, \
                              it is recommended to choose a different order")
                sol[-q:] = np.zeros(q)
        except:
            sol[-q:] = np.zeros(q)
        
        return DataLoader(sol).to_tensor()

    def initialize_parameters(self):
        """Initialize model parameters"""
        self.__init_used_params()

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

        ## TODO: ADD CONSTRAINT INEQUALITIES TO ENSURE STATIONARITY/INVERTIBILITY
        print("Estimating Parameters...")
        print(f"Initial Estimates for coefficients: {init_values}")
        results = opt.minimize(fun=func, x0=init_values,
                                   method="L-BFGS-B")
       
        print(results)
        result_params = list(results.x)
        
        def assign_results(results):
            start_ind = 0
            for param, order in self.used_params.items():
                values = results[start_ind:start_ind+order]
                self.init_params[param] = DataLoader(values).to_tensor()
                start_ind += order

        assign_results(result_params)

        return self.init_params