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
        
        super().__init__(data, **kwargs)              
    
    def __one_step_equation(self, data, errors):
        """
        One step equation for the SARIMA model

        y_t is found by solving the equation;

        phi(B**p)*PHI(B**mP)*delta(B**d)*delta(B**mD)*y_t
        - theta(B**q)*THETA(B**mQ) = 0
        """

        #Assume error is 0 for the unknown next y_t
        new_err = torch.cat((errors, torch.zeros(1)))
        
        def func(x):

            #Assume y_t = x; when the function is solved for x
            #where func(x) = 0, we get our predicted y_t value = x
            new_data = torch.cat((data, DataLoader(x).to_tensor()))

            lhs = self.ar_operator*self.seasonal_ar_operator*self.diff_operator*self.seasonal_diff_operator*new_data
            rhs = self.ma_operator*self.seasonal_ma_operator*new_err

            pred = lhs - rhs

            return DataLoader(pred).to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = opt.fsolve(func, x0=0)[0]

        return DataLoader(sol).to_tensor()
    
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
