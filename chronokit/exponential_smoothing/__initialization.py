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
from chronokit.base._initializers import Initializer

class SmoothingInitializer(Initializer):

    def __init__(self, model, method="mle"):
        """
        Initializer model for Smoothing models
        which fall under InnovationsStateSpace models
        
        We use Chapter 2.6 and Chapter 5 from Hyndman, R. et al. [1]
        as a reference for estimation methods

        Also, we have looked at statsmodels[2] repository for starting
        value choices for smoothing parameters to use in estimation

        [1] 'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        
        [2] "Seabold, Skipper, and Josef Perktold. 
            'statsmodels: Econometric and statistical modeling with python.' 
            Proceedings of the 9th Python in Science Conference. 2010."
        """

        if method not in ["heuristic", "mle", "simple"]:
            warnings.warn(
                    f"{method} is not a valid method as of v1.1.x/n\
                      Will use heuristic initalization.",
                    stacklevel=2,
                )
            method = "heuristic"
        
        super().__init__(model, method)

        self.used_params = {"components": ["level"],
                            "params": ["alpha"]}        
        
        #Can encounter data length related issues
        try:
            init_level, init_trend, init_seasonal = self.__heuristic_initialization()
        except:  # noqa: E722
            init_level = self.model.data[0]
            init_trend = 1. if self.model.trend == "mul" else 0.
            init_seasonal = [0.]
            if self.model.seasonal is not None:
                init_seasonal = [1. if self.model.seasonal == "mul" else 0. for x in range(self.model.seasonal_periods)]
        
        self.init_components = {"level": init_level, 
                                "trend_factor": init_trend,
                                "seasonal_factors": init_seasonal}

        init_alpha = 0.1 if model.seasonal is None else 0.5/model.seasonal_periods
        self.init_params = {"alpha": init_alpha} 

        if method != "heuristic":
            self.err_func = {"mle": self._log_likelihood_err,
                             "simple": self._sse_err}[self.estimation_method]

    def __init_used_params(self):
        
        if self.model.trend is not None:
            self.used_params["components"].append("trend_factor")
            self.used_params["params"].append("beta")
            self.init_params["beta"] = 0.01

            if self.model.damped:
                self.used_params["params"].append("phi")
                self.init_params["phi"] = 0.99
            else:
                self.init_params["phi"] = 1.
        else:
            self.init_params["beta"] = 0.
            self.init_params["phi"] = 1.
        
        if self.model.seasonal is not None:
            self.used_params["components"].append("seasonal_factors")
            self.used_params["params"].append("gamma")
            self.init_params["gamma"] = 0.05*(1-self.init_params["alpha"])
        else:
            self.init_params["gamma"] = 0.

        for component, value in self.init_components.items():
                setattr(self.model, component, value)
        
        for param, value in self.init_params.items():
                setattr(self.model, param, value)

    def __estimation_simple(self):
        
        """
        Parameter estimation for heuristic initialization

        See Hyndman, R. et al. Chapter 5.2
        """

        #LinReg on the parameters only
        def func(x):

            for component, value in self.init_components.items():
                setattr(self.model, component, value)

            for i, param in enumerate(self.used_params["params"]):
                setattr(self.model, param, x[i])
            
            self.model.fit()
            err = self.model.data - self.model.fitted

            return self._sse_err(err)
        
        init_params = [self.init_params[param] for param in self.used_params["params"]]
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results = opt.least_squares(fun=func, x0=init_params, bounds=(1e-6,1-1e-6), max_nfev=10)
            estimated_params = results.x

            init_components = [self.init_components["level"]]
            if "trend_factor" in self.used_params["components"]:
                init_components.append(self.init_components["trend_factor"])
            if "seasonal_factors" in self.used_params["components"]:
                for s in self.init_components["seasonal_factors"]:
                    init_components.append(s)

            result_params = init_components + list(estimated_params)
        except:  # noqa: E722
            return None, False
    
        return result_params, results.success
    
    def __estimation_minimize(self):
        """
        Estimation of parameters, along with the starting
        component values

        Works by minimizing either negative log_likelihood 
        or sum of squared errors

        We also choose initial estimates for component values
        by the heuristic initialization results

        See Hyndman, R. et al. Chapter 5.1 as reference
        """
        init_components = [self.init_components["level"]]
        if "trend_factor" in self.used_params["components"]:
            init_components.append(self.init_components["trend_factor"])
        if "seasonal_factors" in self.used_params["components"]:
            for s in self.init_components["seasonal_factors"]:
                init_components.append(s)

        init_params = [v for k,v in self.init_params.items() if k in self.used_params["params"]]

        init_values = init_components + init_params

        def func(param_values):

            components = param_values[:-len(self.used_params["params"])]
            params = param_values[-len(self.used_params["params"]):]

            if self.model.seasonal is not None:
                seasonal_components = components[-self.model.seasonal_periods:]           
                components = components[:-self.model.seasonal_periods].tolist() + [seasonal_components]

            for i, component in enumerate(self.used_params["components"]):
                setattr(self.model, component, components[i])
        
            for i, param in enumerate(self.used_params["params"]):
                setattr(self.model, param, params[i])

            self.model.fit()
            err = self.model.data - self.model.fitted
            
            return self.err_func(err)

        param_bounds = [(None,None) for i in init_components]
        for param in init_params:
            param_bounds.append((1e-6,1-1e-6))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results = opt.minimize(
                                    func,
                                    init_values,
                                    bounds=param_bounds
                )
        except:  # noqa: E722
            return None, False


        return list(results.x), results.success
        
    def __heuristic_initialization(self):
        """
        Heuristic initialization method for initial components
        See: Hyndman, R. et al. Chapter 2.6.1 and Chapter 5.2 
        """

        data = DataLoader(self.model.data).to_numpy().copy()
        data = DataLoader(data).match_dims(dims=1, return_type="numpy")

        trend = self.model.trend
        seasonal = self.model.seasonal
        seasonal_periods = self.model.seasonal_periods

        initial_level = 0.
        initial_trend = 0.
        initial_seasonal = [0.]

        assert len(data) >= 10, "Length of data must be >= for heuristic initialization"

        if seasonal:
            # Data must have at least 2 full seasonal cycles
            assert len(data) > 2 * seasonal_periods, "Length of data must be > 2*seasonal_periods"

            # Max number of seasonal cycles to be used is 5
            seasonal_cycles = min(5, len(data) // seasonal_periods)
            series = pd.Series(data[: seasonal_periods * seasonal_cycles])
            moving_avg = series.rolling(seasonal_periods, center=True).mean()

            if seasonal_periods % 2 == 0:
                moving_avg = moving_avg.shift(-1).rolling(2).mean()

            if seasonal == "add":
                detrend = series - moving_avg
            elif seasonal == "mul":
                detrend = series / moving_avg

            initial_seasonal = np.zeros(shape=(seasonal_periods, seasonal_cycles)) * np.nan
            for i in range(0, seasonal_cycles):
                initial_seasonal[:, i] = detrend[i * seasonal_periods : (i + 1) * seasonal_periods]

            initial_seasonal = np.nanmean(initial_seasonal, axis=1)

            if seasonal == "add":
                # normalize so that the sum is equal to 0
                initial_seasonal -= np.mean(initial_seasonal)
            elif seasonal == "mul":
                # normalize so that the sum is equal to m=seasonal_periods
                initial_seasonal *= seasonal_periods / np.sum(initial_seasonal)
            
            initial_seasonal = torch.tensor(initial_seasonal, dtype=torch.float32)

            adjusted_data = moving_avg.dropna().values

        else:
            adjusted_data = data.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = linregress(x=np.arange(1,11), y=adjusted_data[:10])
        initial_level = torch.tensor(result[1], dtype=torch.float32)

        if trend == "add":
            initial_trend = torch.tensor(result[0], dtype=torch.float32)
        elif trend == "mul":
            initial_trend = 1 + torch.tensor(result[0], dtype=torch.float32)/initial_level

        return initial_level, initial_trend, initial_seasonal
    
    def initialize_parameters(self):
        """Initialize model parameters"""
        self.__init_used_params()

        if self.estimation_method != "heuristic":
            result_params, self.success = self.__estimation_minimize()
        
        else:
            result_params, self.success = self.__estimation_simple()
        
        def assign_results(results):
            
            self.init_components["level"] = DataLoader(results[0]).to_tensor()

            has_trend = False
            if "trend_factor" in self.used_params["components"]:
                self.init_components["trend_factor"] = DataLoader(results[1]).to_tensor()
                has_trend = True

            if "seasonal_factors" in self.used_params["components"]:
                start_ind = 2 if has_trend else 1
                self.init_components["seasonal_factors"] = DataLoader(results[start_ind:self.model.seasonal_periods+start_ind]).to_tensor()
            
            param_results = results[-len(self.used_params["params"]):]

            for i, param in enumerate(self.used_params["params"]):
                self.init_params[param] = param_results[i]
        
        if result_params is not None:
            assign_results(result_params)

        return self.init_params, self.init_components