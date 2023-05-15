import numpy as np
import torch
from scipy.optimize import least_squares
import pandas as pd
from dataloader import DataLoader
import ets_methods
from initialization_methods import heuristic_initialization
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

    

class ETS_Model(Model):

    def __init__(self, dep_var, trend=None,  seasonal=None, error="add", seasonal_periods=None, damped=False, indep_var=None, **kwargs):
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma", "conf"])
        super().__init__(dep_var, indep_var, **kwargs)

        #keys are [trend, damped, seasonal, error(add or mul)]
        self.method = { 
                        (None, False, None, "add"): None,
                        (None, False, "add", "add"): None,
                        (None, False, "mul", "add"): None,
                        ("add", False, None, "add"): None,
                        ("add", False, "add", "add"): ets_methods.AAA,
                        ("add", False, "mul", "add"): ets_methods.AAM,
                        ("add", True, None, "add"): None,
                        ("add", True, "add", "add"): None,
                        ("add", True, "mul", "add"): ets_methods.AAdM,

                        (None, False, None, "mul"): None,
                        (None, False, "add", "mul"): None,
                        (None, False, "mul", "mul"): None,
                        ("add", False, None, "mul"): None,
                        ("add", False, "add", "mul"): ets_methods.MAA,
                        ("add", False, "mul", "mul"): None,
                        ("add", True, None, "mul"): None,
                        ("add", True, "add", "mul"): None,
                        ("add", True, "mul", "mul"): ets_methods.MAdM
                                                            }[trend, damped, seasonal, error]
        

    def estimate_params(self, init_components, params):

        def func(x):

            errs = self.method(self.dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = torch.tensor(least_squares(fun=func, x0 = params, bounds=(0,1)).x)

        return estimated_params

    def calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""

        return round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)
    




