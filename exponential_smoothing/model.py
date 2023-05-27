import numpy as np
import torch
from scipy.optimize import least_squares
import pandas as pd
from dataloader import DataLoader
from .initialization import ets_methods
from .initialization import smoothing_methods
from .initialization.initialization_methods import heuristic_initialization
import scipy.stats as stats

class Model():
    def __init__(self,dep_var, indep_var=None,**kwargs):
        """"A template class that provides a skeleton for model classes that
            to inherit from.
            Child classes are exppected to implement their own .fit() and .predict() methods
            """
        
        self.dep_var = DataLoader(dep_var).to_tensor()
        if indep_var is not None:
            self.indep_var = DataLoader(indep_var).to_tensor()
        self.set_kwargs(kwargs)

    
    def set_allowed_kwargs(self, kwargs: list):
        '''This function sets the allowed keyword arguments for the model.'''
        self.allowed_kwargs = kwargs


    def __check_kwargs(self, kwargs: dict):
          '''This function checks if the keyword arguments are valid.'''
          for (k,v) in kwargs.items():
              if k not in self.allowed_kwargs:
                  raise ValueError("{key} is not a valid keyword for this model".format(key = k))


    def set_kwargs(self, kwargs: dict):
        '''This function sets the keyword arguments for the model.'''
        self.__check_kwargs(kwargs)
        for (k,v) in kwargs.items():
            self.__setattr__(k,v)

    def fit(self, dep_var, indep_var, *args, **kwargs):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, dep_var, indep_var, *args, **kwargs):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")
    
class Smoothing_Model(Model):

    def __init__(self, dep_var, trend=None,  seasonal=None,  seasonal_periods=None, damped=False, indep_var=None, **kwargs):
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, indep_var, **kwargs)

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        #keys are [trend, damped, seasonal, error(add or mul)]
        self.method = { 
                        (None, False, None): smoothing_methods.simple_exp,
                        ("add", False, None): smoothing_methods.holt_trend,
                        ("add", True, None): smoothing_methods.holt_damped_trend,
                        ("add", False, "add"): smoothing_methods.hw_add,
                        ("add", False, "mul"): smoothing_methods.hw_mul,
                        ("add", True, "add"): smoothing_methods.hw_damped_add,
                        ("add", True, "mul"): smoothing_methods.hw_damped_mul
                                                            }[trend, damped, seasonal]
        
        self.params = {"alpha": 0.1, "beta": 0.01, "gamma": 0.01, "phi": 0.99}
        self.init_components = {"level": None, "trend": None, "seasonal": None}
        

    def __estimate_params(self, init_components, params):

        def func(x):

            errs = self.method(self.dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = least_squares(fun=func, x0 = params, bounds=(0,1)).x

        return estimated_params
    
    def initialize_params(self, initialize_params):

        self.initial_level, self.initial_trend, self.initial_seasonals = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

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


class ETS_Model(Model):

    def __init__(self, dep_var, trend=None,  seasonal=None, error_type="add", seasonal_periods=None, damped=False, indep_var=None, **kwargs):
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma", "conf"])
        super().__init__(dep_var, indep_var, **kwargs)

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.error_type = error_type

        #keys are [trend, damped, seasonal, error(add or mul)]
        self.method = { 
                        (None, False, None, "add"): ets_methods.ANN,
                        (None, False, "add", "add"): ets_methods.ANA,
                        (None, False, "mul", "add"): ets_methods.ANM,
                        ("add", False, None, "add"): ets_methods.AAN,
                        ("add", False, "add", "add"): ets_methods.AAA,
                        ("add", False, "mul", "add"): ets_methods.AAM,
                        ("add", True, None, "add"): ets_methods.AAdN,
                        ("add", True, "add", "add"): ets_methods.AAdA,
                        ("add", True, "mul", "add"): ets_methods.AAdM,

                        (None, False, None, "mul"): ets_methods.MNN,
                        (None, False, "add", "mul"): ets_methods.MNA,
                        (None, False, "mul", "mul"): ets_methods.MNM,
                        ("add", False, None, "mul"): ets_methods.MAN,
                        ("add", False, "add", "mul"): ets_methods.MAA,
                        ("add", False, "mul", "mul"): ets_methods.MAM,
                        ("add", True, None, "mul"): ets_methods.MAdN,
                        ("add", True, "add", "mul"): ets_methods.MAdA,
                        ("add", True, "mul", "mul"): ets_methods.MAdM
                                                            }[trend, damped, seasonal, error_type]
        
        self.params = {"alpha": 0.1, "beta": 0.01, "gamma": 0.01, "phi": 0.99}
        self.init_components = {"level": None, "trend": None, "seasonal": None}
        

    def __estimate_params(self, init_components, params):

        def func(x):

            errs = self.method(self.dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = least_squares(fun=func, x0 = params, bounds=(0,1)).x

        return estimated_params
    
    def initialize_params(self, initialize_params):

        self.initial_level, self.initial_trend, self.initial_seasonals = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

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

        """Calculate the confidence level to be used for intervals"""

        return round(stats.norm.ppf(1 - ((1 - conf) / 2)), 2)
    
    def update_res_variance(self, residuals, error):

        residuals = torch.cat((residuals, torch.reshape(error, (1,1))))
        
        res_mean = torch.sum(residuals)/residuals.shape[0]

        residual_variance = torch.sum(torch.square(torch.sub(residuals, res_mean)))
        residual_variance = torch.divide(residual_variance, residuals.shape[0]-1)

        return residuals, res_mean, residual_variance
    
    def future_sample_paths(self, h, confidence):
        
        q1 = (1-confidence)/2
        q2 = 1 - q1

        loc = self.residual_mean
        scale = self.residual_variance

        sample_paths = torch.tensor([])

        for iter in range(5000):

            sample = torch.normal(loc, scale, size=(1,h))
            sample_paths = torch.cat((sample_paths, sample))

        q1_sample = torch.quantile(sample_paths, q1, dim=0, interpolation="nearest")
        q2_sample = torch.quantile(sample_paths, q2, dim=0, interpolation="nearest")

        bounds = torch.abs(torch.sub(q1_sample, q2_sample))

        return bounds




