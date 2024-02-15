import numpy as np
import torch
import warnings
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader

class TraditionalTimeSeriesModel:

    def __init__(self, data, **kwargs):
        """
        Base model class for all model classes to inherit from.
        Child classes are expected to implement their own .fit()
        and .predict() methods
        """

        self.data_loader = DataLoader(data)

        #Assert Valid Data
        try:
            self.data = self.data_loader.match_dims(1, return_type="torch")
        except:  # noqa: E722
            raise NotImplementedError(
                "Multivariate models are not implemented as of v1.1.0,\
                please make sure data.ndim <= 1 or squeezable"
            )

        self.info = {}

    def report_fit_info(self):
        pass


    def _confidence_interval_simulation_method(self, h, confidence):
        """Refer to Hyndman et al. Chapter 6.1"""
        print("called")
        M = 5000 #AS recommended by Hyndman et. al
        q1 = (1-confidence)/2
        q2 = 1 - q1

        fit_errors = self.data - self.fitted

        #torch.nanstd() has not been implemented yet
        #we calculate in numpy then convert to torch tensors
        loc = torch.tensor(np.nanmean(fit_errors.clone().numpy()), dtype=torch.float32)
        scale = torch.tensor(np.nanstd(fit_errors.clone().numpy()), dtype=torch.float32)

        sample_errors = torch.normal(loc, scale, size=(M,h))

        q1_sample = torch.quantile(sample_errors, q1, dim=0, interpolation="nearest")
        q2_sample = torch.quantile(sample_errors, q2, dim=0, interpolation="nearest")

        bounds = torch.abs(torch.sub(q1_sample, q2_sample))

        return bounds
    
    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        valid_kwargs = self.check_kwargs(kwargs)
        for k, v in valid_kwargs.items():
            self.__setattr__(k, v)
    
    def check_kwargs(self, kwargs: dict):
        return kwargs
    
    def fit(self):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, h: int, confidence: float = None):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

