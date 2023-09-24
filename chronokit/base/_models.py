import numpy as np
import torch
from scipy.optimize import least_squares
import pandas as pd
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.exponential_smoothing.initialization import get_init_method, get_smooth_method
import scipy.stats as stats

class Model():
    def __init__(self, data, **kwargs):
        """
        Base model class for all model classes to inherit from.
        Child classes are expected to implement their own .fit() and .predict() methods
        """
        
        self.data = DataLoader(data).to_tensor()
        self.set_kwargs(kwargs)

        if self.data.ndim == 2:

            if min(self.data.shape) != 1:
                raise NotImplementedError("Multivariate models are not implemented as of v1.0.0, please make sure data.ndim <= 1 or squeezable")
            else:
                self.data = self.data.squeeze(torch.tensor(self.data.shape).argmin())
        
        elif self.data.ndim > 2:
            raise NotImplementedError("Multivariate models are not implemented as of v1.0.0, please make sure data.ndim <= 1 or squeezable")
    
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




