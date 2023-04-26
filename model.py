import numpy as np
import pandas as pd
from dataloader import DataLoader

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



