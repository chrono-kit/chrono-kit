import torch
import numpy as np
import warnings
import scipy.optimize as opt
from scipy.linalg import toeplitz
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.preprocessing import differencing
from chronokit.preprocessing.autocorrelations import AutoCorrelation
from chronokit.arima.__base_model import ARIMAProcess
from chronokit.base._models import TraditionalTimeSeriesModel
from chronokit.arima.__initialization import SARIMAInitializer

"""
SARIMA Models for time series analysis and forecasting

SARIMA Models was developed by the below book as a reference;

'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
Time series analysis: Forecasting and control (5th ed).'

General ARIMA equation;

(1 - phi_1*B - ... - phi_p*(B**p))((1-B)**d)*y_t = 
(1 - theta_1*B - ... - theta_q*(B**q))*e_t

This equation reduces to;

* ARMA(p,q) : If d = 0
* AR(p) : If d = 0 and q = 0
* MA(q): If d = 0 and p = 0
* ARI(p,d): If q = 0
* IMA(d,q) : If p = 0

We can also denote the equations as;

phi(B**p) = (1 - phi_1*B - ... - phi_p*(B**p))
delta(d) = (1-B)**d
theta(B**q) = (1 - theta_1*B - ... - theta_q*(B**q)

Hence;

ARIMA(p,d,q) : phi(B**p)*delta(d)*y_t = theta(B**q)*e_t

Seasonal ARIMA (SARIMA) is "intuitively" an ARIMA model that "jumps" by m
in model equations, where m is the seasonality period

SARIMA models have order {(p,d,q), (P,D,Q)m} where;

(p,d,q): Order of a normal ARIMA model
(P,D,Q): Order of the seasonal model
m: Seasonality period

SARIMA(0,0,0),(P,D,Q)m : PHI(B**(m*P))*delta(m*D)*y_t = THETA(m*Q)*e_t

where PHI, THETA are the same as phi(B), theta(B) operators above 
but with different parameter values

Hence, A SARIMA(p,d,q),(P,D,Q)_m model is denoted as;


(1 -...- phi_p*(B**p))*(1 -...- PHI_P*(B**(m*P))*((1-B)**d)*((1-B)**(m*D)*y_t = 
(1 -... - theta_q*(B**q))*(1 -...- THETA_Q*(B**(m*Q))*e_t

or


phi(B**p)*PHI(B**mP)*delta(d)*delta(m*D)*y_t = theta(B**q)*THETA(B**mD)*e_t

For further reading refer to;

Chapter 4-5 of the book Box, G. et al. for ARMA and ARIMA models
Chapter 9 of the book Box, G. et al. for Seasonal ARIMA models

"""

class SARIMA(ARIMAProcess):
    def __init__(
                self, 
                data, 
                order = (0,0,0), 
                seasonal_order=(0,0,0),
                seasonal_periods=None,
                initialization_method="simple",
                **kwargs
):
        
        """
        Seasonal AutoRegressive-Moving Average Integrated Model (SARIMA)
        for univarite time series modeling and forecasting.

        Arguments:

        *data (array_like): Univariate time series data
        *order (Optional[tuple]): Order of the ARIMA process (p,d,q)
        *seasonal_order (Optional[tuple]): Order of the seasonal process (P,D,Q)
            (0,0,0) if seasonal_periods is None
        *seasonal_periods (Optional[int]): Cyclic period of the
            seasonal component; None if seasonal_order = (0,0,0)

        Keyword Arguments:

        ** phi (array_like): Parameters for AR component;
            must be of length p
        ** theta (array_like): Parameters for MA component;
            must be of length q
        ** seasonal_phi (array_like): Parameters for Seasonal AR component;
            must be of length P
        ** seasonal_theta (array_like): Parameters for Seasonal MA component;
            must be of length Q

        SARIMA Model was developed by the below book as a reference;

        'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
        Time series analysis: Forecasting and control (5th ed).'
        """
        
        super().set_allowed_kwargs(["phi", "theta",
                                    "seasonal_phi",
                                    "seasonal_theta"])

        self.order = order
        self.p, self.d, self.q = order
        self.seasonal_order = seasonal_order
        self.P, self.D, self.Q = seasonal_order
        self.seasonal_periods = seasonal_periods
        
        super().__init__(data, initialization_method, **kwargs)
    
    def fit(self):
        """Fit the model to the provided univariate time series data"""

        #Ensure valid parameters and operations
        self.__prepare_operations__()

        self.phi = DataLoader(self.phi).to_tensor()
        self.theta = DataLoader(self.theta).to_tensor()
        self.seasonal_phi = DataLoader(self.seasonal_phi).to_tensor()
        self.seasonal_theta = DataLoader(self.seasonal_theta).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        self.errors = torch.zeros(size=self.data.shape)
        
        lookback = len(self.w_mat)

        for ind, y in enumerate(self.data):
            
            cur_data = self.data[:ind].clone()
            if ind < lookback:
                cur_data = torch.cat((torch.zeros(lookback-ind), cur_data))
            elif ind > lookback:
                cur_data = self.data[ind-lookback:ind].clone()
            
            y_hat = torch.sum(self.w_mat*cur_data)

            self.fitted[ind] = y_hat
            self.errors[ind] = y - y_hat     
    
    def predict(self, h, confidence=None):
        """
        Predict the next h values with the model

        .fit() should be called before making predictions

        Arguments:

        *h (int): Forecast horizon, minimum = 1
        *confidence (Optional[float]): Confidence bounds
            for predictions. Must take values in (0,1).
        
        Returns:

        *forecasts (array_like): Forecasted h values
        *bounds Optional(tuple): Upper and Lower
            forecast bounds, returns only if confidence != None
        """
        
        forecasts = torch.zeros(h)

        lookback = len(self.w_mat)
        cur_data = self.data[-lookback:].clone()
        if len(cur_data) < lookback:
            cur_data = torch.cat((torch.zeros(lookback-len(cur_data)), cur_data))

        for step in range(h):
            
            y_hat = torch.sum(self.w_mat*cur_data)

            cur_data = torch.cat((cur_data[1:], DataLoader(y_hat).match_dims(1, return_type="tensor")))
        
            forecasts[step] = y_hat
        
        if confidence is not None:
            if self.data_loader.is_valid_numeric(confidence) and (confidence < 1 and confidence > 0):
                upper_bounds, lower_bounds = self.calculate_confidence_interval(forecasts, float(confidence))

                return forecasts, (upper_bounds, lower_bounds)
        
        return forecasts
