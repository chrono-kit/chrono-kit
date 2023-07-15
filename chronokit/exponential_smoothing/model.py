import numpy as np
import torch
from scipy.optimize import least_squares
import pandas as pd
from chronokit.preprocessing.dataloader import DataLoader
from .initialization import get_init_method, get_smooth_method
import scipy.stats as stats

class Model():
    def __init__(self,dep_var, **kwargs):
        """
        Base model class for all model classes to inherit from.
        Child classes are expected to implement their own .fit() and .predict() methods
        """
        
        self.dep_var = DataLoader(dep_var).to_tensor()
        self.set_kwargs(kwargs)

    
    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs


    def __check_kwargs(self, kwargs: dict):
          """This function checks if the keyword arguments are valid."""
          for (k,v) in kwargs.items():
              if k not in self.allowed_kwargs:
                  raise ValueError("{key} is not a valid keyword for this model".format(key = k))

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        self.__check_kwargs(kwargs)
        for (k,v) in kwargs.items():
            self.__setattr__(k,v)

    
class Smoothing_Model(Model):

    def __init__(self, dep_var, trend=None,  seasonal=None,  seasonal_periods=None, damped=False, initialization_method="heuristic", **kwargs):
        """
        Base class for exponential smoothing methods.
        All smoothing methods inherit from this class and is used for parameter initialization.
        
        Arguments:

        *dep_var (array_like): Univariate time series data
        *trend (Optional[str]): Trend component; None or "add"
        *seasonal (Optional[str]): Seasonal component; None, "add" or "mul"
        *seasonal_periods (Optional[int]): Cyclic period of the seasonal component; int or None if seasonal is None
        *damped (bool): Damp factor of the trend component; False if trend is None
        *initialization_method (str): Initialization method to use for the model parameters; "heuristic" or "mle"

        Keyword Arguments:

        ** alpha (float): Smoothing parameter for level component; takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component; takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component; takes values in (0,1)
        
        All of the smoothing methods have been written as the below textbook as a reference:
        'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.'
        """
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, **kwargs)

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.init_method = initialization_method

        self.method = get_smooth_method(error=None, trend=trend, damped=damped, seasonal=seasonal)
        
        self.params = {"alpha": 0.1, "beta": 0.01, "gamma": 0.01, "phi": 0.99}
        self.init_components = {"level": None, "trend": None, "seasonal": None}
        

    def __estimate_params(self, init_components, params):
        """Estimate the best parameters to use during fitting and forecasting for the smoothing model"""
        def func(x):
            if self.dep_var.ndim == 1:
                dep_var = torch.unsqueeze(self.dep_var, axis=-1)
            else:
                dep_var = self.dep_var
            errs = self.method(dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = least_squares(fun=func, x0 = params, bounds=(0,1)).x

        return estimated_params
    
    def initialize_params(self, initialize_params):
        """Initialize the components and the parameters to use during fitting and forecasting for the smoothing model"""
        self.initial_level, self.initial_trend, self.initial_seasonals = get_init_method(method=self.init_method)(self.dep_var,trend=self.trend, 
                                                                                                                  seasonal=self.seasonal, 
                                                                                                                  seasonal_periods=self.seasonal_periods)

        self.init_components["level"] = self.initial_level,
        self.init_components["trend"] = self.initial_trend,
        self.init_components["seasonal"] = np.expand_dims(self.initial_seasonals, axis=-1)

        init_params = {param: self.params[param] for param in initialize_params}

        params = self.__estimate_params(
                        init_components = [self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods],
                        params = list(init_params.values()))
        
        for index, param in enumerate(params):

            init_params[list(init_params.keys())[index]] = param

        for param in self.params:

            if param in init_params:
                self.params[param] = init_params[param]

        if not self.damped:
            self.params["phi"] = 1
    
    def fit(self):
        #This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, h: int):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")


class ETS_Model(Model):

    def __init__(self, dep_var, error_type="add", trend=None,  seasonal=None, seasonal_periods=None, damped=False, initialization_method="heuristic", **kwargs):
        """
        Base class for ETS models
        All smoothing methods inherit from this class and is used for parameter initialization.
        
        Arguments:

        *dep_var (array_like): Univariate time series data
        *error_type (str): Type of error of the ETS model; "add" or "mul"
        *trend (Optional[str]): Trend component; None or "add"
        *seasonal (Optional[str]): Seasonal component; None, "add" or "mul"
        *seasonal_periods (Optional[int]): Cyclic period of the seasonal component; int or None if seasonal is None
        *damped (bool): Damp factor of the trend component; False if trend is None
        *initialization_method (str): Initialization method to use for the model parameters; "heuristic" or "mle"

        Keyword Arguments:

        ** alpha (float): Smoothing parameter for level component; takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component; takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component; takes values in (0,1)
        
        All of the smoothing methods have been written as the below textbook as a reference:
        'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.'
        """
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var,  **kwargs)

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.error_type = error_type
        self.init_method = initialization_method

        self.method = get_smooth_method(error=error_type, trend=trend, damped=damped, seasonal=seasonal)

        self.params = {"alpha": 0.1, "beta": 0.01, "gamma": 0.01, "phi": 0.99}
        self.init_components = {"level": None, "trend": None, "seasonal": None}
        

    def __estimate_params(self, init_components, params):
        """Estimate the best parameters to use during fitting and forecasting for the smoothing model"""
        def func(x):
            if self.dep_var.ndim == 1:
                dep_var = torch.unsqueeze(self.dep_var, axis=-1)
            else:
                dep_var = self.dep_var
            errs = self.method(dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = least_squares(fun=func, x0 = params, bounds=(0,1)).x

        return estimated_params
    
    def initialize_params(self, initialize_params):
        """Initialize the components and the parameters to use during fitting and forecasting for the smoothing model"""
        self.initial_level, self.initial_trend, self.initial_seasonals = get_init_method(method=self.init_method)(self.dep_var,trend=self.trend, 
                                                                                                                  seasonal=self.seasonal, 
                                                                                                                  seasonal_periods=self.seasonal_periods)
        self.init_components["level"] = self.initial_level,
        self.init_components["trend"] = self.initial_trend,
        self.init_components["seasonal"] = np.expand_dims(self.initial_seasonals, axis=-1)

        init_params = {param: self.params[param] for param in initialize_params}

        params = self.__estimate_params(
                        init_components = [self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods],
                        params = list(init_params.values()))
        
        for index, param in enumerate(params):

            init_params[list(init_params.keys())[index]] = param

        for param in self.params:

            if param in init_params:
                self.params[param] = init_params[param]

        if not self.damped:
            self.params["phi"] = 1

    def calculate_conf_level(self, conf):
        """Calculate the confidence level to be used for prediction intervals"""

        return round(stats.norm.ppf(1 - ((1 - conf) / 2)), 2)
    
    def update_res_variance(self, residuals, error):
        """Update the variance of the residuals during fitting"""
        residuals = torch.cat((residuals, torch.reshape(error, (1,1))))
        res_mean = torch.sum(residuals)/residuals.shape[0]
        residual_variance = torch.sum(torch.square(torch.sub(residuals, res_mean)))
        residual_variance = torch.divide(residual_variance, residuals.shape[0]-1)

        return residuals, res_mean, residual_variance
    
    def future_sample_paths(self, h, confidence):
        """
        Future path sampling for ETS models with no known equations for generating prediction intervals
        Errors are assumed to be normally distributed and random future paths are sampled from the normal distribution
        with mean and variance calculated by the residuals during fitting 
        """
        q1 = (1-confidence)/2
        q2 = 1 - q1

        loc = self.residual_mean
        scale = torch.sqrt(self.residual_variance)

        sample_paths = torch.tensor([])

        for iter in range(5000):

            sample = torch.normal(loc, scale, size=(1,h))
            sample_paths = torch.cat((sample_paths, sample))

        q1_sample = torch.quantile(sample_paths, q1, dim=0, interpolation="nearest")
        q2_sample = torch.quantile(sample_paths, q2, dim=0, interpolation="nearest")

        bounds = torch.abs(torch.sub(q1_sample, q2_sample))

        return bounds
    
    def fit(self):
        #This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, h: int, confidence: float = None):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")




