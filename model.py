import numpy as np
import pandas as pd


class Model():
    def __init__(self,dep_var, indep_var,**kwargs):
        self.data = pd.DataFrame()
        self.data.dep_var = dep_var
        self.data.indep_var = indep_var

        if "allowed_kwargs" in kwargs:
            self.__allowed_kwargs = kwargs["allowed_kwargs"]
            self.__setkwargs(kwargs)

    def __checkkwargs(self, kwargs: dict):
          for (k,v) in kwargs.items():
              if k not in self.__allowed_kwargs:
                  raise ValueError("{key} is not a valid keyword for this model".format(key = k))


    def __setkwargs(self, kwargs: dict):
        self.__checkkwargs(kwargs)
        for (k,v) in kwargs.items():
            self.__setattr__(k,v)

    def fit(self, dep_var, indep_var, *args, **kwargs):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, dep_var, indep_var, *args, **kwargs):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")



