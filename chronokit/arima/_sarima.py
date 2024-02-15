import torch
import numpy as np
from chronokit.arima.__base_model import ARIMAProcess
from chronokit.preprocessing.autocorrelations import AutoCorrelation
import scipy.optimize as opt
from scipy.linalg import toeplitz
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.preprocessing import differencing
import warnings

warnings.filterwarnings("ignore")

"""
ARIMA models for time series forecasting. All methods have been
implemented from chapter 8 of the textbook as a reference.
'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
and practice. OTexts, 2014.'
"""

class SARIMA(ARIMAProcess):
    def __init__(self, data, 
                    order = (0,0,0), 
                    seasonal_order=(0,0,0),
                    seasonal_periods=None,
                    **kwargs):
        
        super().set_allowed_kwargs(["phi", "theta",
                                    "seasonal_phi",
                                    "seasonal_theta"])

        self.order = order
        self.p, self.d, self.q = order
        self.seasonal_order = seasonal_order
        self.P, self.D, self.Q = seasonal_order
        self.seasonal_periods = seasonal_periods
        
        super().__init__(data, **kwargs)              
    
    def __one_step_equation(self, data, errors):

        new_err = torch.cat((errors, torch.zeros(1)))
        
        def func(x):

            new_data = torch.cat((data, DataLoader(x).to_tensor()))

            lhs = self.ar_operator*self.seasonal_ar_operator*self.diff_operator*self.seasonal_diff_operator*new_data
            rhs = self.ma_operator*self.seasonal_ma_operator*new_err

            pred = lhs - rhs

            return DataLoader(pred).to_numpy()
    
        sol = opt.fsolve(func, x0=0)[0]

        return DataLoader(sol).to_tensor()
    
    def fit(self):

        self.__prepare_operations__()

        self.phi = DataLoader(self.phi).to_tensor()
        self.theta = DataLoader(self.theta).to_tensor()
        self.seasonal_phi = DataLoader(self.seasonal_phi).to_tensor()
        self.seasonal_theta = DataLoader(self.seasonal_theta).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        self.errors = torch.zeros(size=self.data.shape)

        m = 0 if self.seasonal_periods is None else self.seasonal_periods
        lookback = self.p + self.d + m*(self.P + self.D) 
        err_lookback = self.q + m*self.Q

        for ind, y in enumerate(self.data):
            if ind < lookback:
                self.fitted[ind] = torch.nan
                continue

            cur_data = self.data[ind-lookback:ind].clone()
            
            if ind < err_lookback:
                cur_errors = torch.zeros(err_lookback)
            else:
                cur_errors = self.errors[ind-err_lookback:ind].clone()
            
            y_hat = self.__one_step_equation(cur_data, cur_errors)

            self.fitted[ind] = y_hat
            self.errors[ind] = y - y_hat     
    
    def predict(self, h, confidence=None):

        forecasts = torch.zeros(h)

        m = 0 if self.seasonal_periods is None else self.seasonal_periods
        lookback = self.p + self.d + m*(self.P + self.D)
        err_lookback = self.q + m*self.Q
        cur_data = self.data[-lookback:].clone()
        cur_errors = self.errors[-err_lookback:].clone()

        for step in range(h):
            
            y_hat = self.__one_step_equation(cur_data, cur_errors)

            cur_data = torch.cat((cur_data[1:], DataLoader(y_hat).match_dims(1, return_type="tensor")))
            cur_errors = torch.cat((cur_errors[1:], torch.zeros(1)))

            forecasts[step] = y_hat
        
        if confidence is not None:
            if self.data_loader.is_valid_numeric(confidence) and (confidence < 1 and confidence > 0):
                upper_bounds, lower_bounds = self.calculate_confidence_interval(forecasts, float(confidence))

                return forecasts, (upper_bounds, lower_bounds)
        
        return forecasts
