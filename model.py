import numpy as np
import torch
from scipy.optimize import least_squares
import pandas as pd
from dataloader import DataLoader
import ets_methods
from initialization_methods import heuristic_initialization

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

    def estimate_params(self, init_components, params):
        
        #keys are [trend, damped, seasonal, error(add or mul)]
        methods = { ("add", True, "mul", "add") : ets_methods.AAdM,
                  }
        
        method = methods[(self.trend, self.damped, self.seasonal, self.error_type)]

        def func(x):

            errs = method(self.dep_var, init_components=init_components, params=x)

            return np.mean(np.square(np.array(errs)))

        estimated_params = torch.tensor(least_squares(fun=func, x0 = params, bounds=(0,1)).x)

        return estimated_params
    




