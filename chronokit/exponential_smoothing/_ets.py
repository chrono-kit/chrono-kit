import torch
import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.exponential_smoothing.__base_model import  InnovationsStateSpace

class ETS(InnovationsStateSpace):
    def __init__(
        self,
        data,
        error_type = None,
        trend=None,
        seasonal=None,
        seasonal_periods=None,
        damped=False,
        initialization_method="heuristic",
        **kwargs,
    ):
        """
        Base class for exponential smoothing methods.hh                             
        All smoothing methods inherit from this class and
        is used for parameter initialization.

        Arguments:

        *data (array_like): Univariate time series data
        *trend (Optional[str]): Trend component; None or "add"
        *seasonal (Optional[str]): Seasonal component; None, "add" or "mul"
        *seasonal_periods (Optional[int]): Cyclic period of the
            seasonal component; int or None if seasonal is None
        *damped (bool): Damp factor of the trend component;
            False if trend is None
        *initialization_method (Optional[str]): Initialization method to use
            for the model parameters; "heuristic", "simple" or "mle". Default heuristic

        Keyword Arguments:

        ** alpha (float): Smoothing parameter for level component;
            takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component;
            takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component;
            takes values in (0,1)

        Smoothing methods have been written with the below book
            as reference: 'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        """
        super().set_allowed_kwargs(["initial_level", 
                                    "initial_trend_factor",
                                    "initial_seasonal_factors",
                                    "alpha", "beta", "phi", "gamma"])

        self.error_type = error_type
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        super().__init__(data, initialization_method, **kwargs)

    def update_state(self, e_t):

        lprev, bprev = self.level.clone(), self.trend_factor.clone()
        s_prev = self.seasonal_factors[0].clone()

        #This is equivalent to passing e_t as additive error
        #However using this approach leads to better parameter estimations
        #This is probably due to usage of scipy.optimize
        if self.error_type == "mul":
            error_term = lprev.clone()

            if self.trend == "mul":
                error_term *= bprev**self.phi
            elif self.trend == "add":
                error_term += bprev*self.phi

            if self.seasonal == "mul":
                error_term *= s_prev
            elif self.seasonal == "add":
                error_term += s_prev
            
            error_term = error_term*e_t
        
        else:
            error_term = e_t.clone()

        if self.seasonal == "mul":
            level_update_p1 = self.alpha*(error_term/s_prev)
            
            if self.trend == "mul":
                seasonal_update_p1 = self.gamma*error_term/(lprev*(bprev**self.phi))
                trend_update_p1 = self.alpha*self.beta*error_term/(s_prev*lprev)
                level_update_p2 = lprev*(bprev**self.phi)

            elif self.trend == "add":
                seasonal_update_p1 = self.gamma*error_term/(lprev + bprev*self.phi)
                trend_update_p1 = self.alpha*self.beta*error_term/s_prev
                level_update_p2 = lprev + bprev*self.phi
            
            else:
                seasonal_update_p1 = self.gamma*error_term/lprev
                level_update_p2 = lprev
        
        elif self.seasonal == "add":
            level_update_p1 = self.alpha*error_term
            seasonal_update_p1 = self.gamma*error_term

            if self.trend == "mul":
                trend_update_p1 = self.alpha*self.beta*error_term/lprev
                level_update_p2 = lprev*(bprev**self.phi)

            elif self.trend == "add":
                trend_update_p1 = self.alpha*self.beta*error_term
                level_update_p2 = lprev + bprev*self.phi
            
            else:
                level_update_p2 = lprev
        
        else:
            level_update_p1 = self.alpha*error_term
            
            if self.trend == "mul":
                trend_update_p1 = self.alpha*self.beta*error_term/lprev
                level_update_p2 = lprev*(bprev**self.phi)

            elif self.trend == "add":
                trend_update_p1 = self.alpha*self.beta*error_term
                level_update_p2 = lprev + bprev*self.phi
            
            else:
                level_update_p2 = lprev
         
        self.level = level_update_p1 + level_update_p2

        if self.trend == "mul":
            self.trend_factor = bprev**self.phi + trend_update_p1
        elif self.trend == "add":
            self.trend_factor = bprev*self.phi + trend_update_p1

        if self.seasonal is not None:
            seasonal = seasonal_update_p1 + s_prev
            self.seasonal_factors = torch.cat((self.seasonal_factors[1:], torch.tensor([seasonal])), dim=-1)
    
    def fit(self):
        
        self.seasonal_factors = DataLoader(self.seasonal_factors).to_tensor()
        self.level = DataLoader(self.level).to_tensor()
        self.trend_factor = DataLoader(self.trend_factor).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        self.errors = torch.zeros(size=self.data.shape)

        for ind, y in enumerate(self.data):
            if ind == 0:
                self.fitted[0] = torch.nan
                self.errors[0] = torch.nan
                continue

            y_hat = self.predict(1)
            self.fitted[ind] = y_hat

            e_t = y - y_hat

            if self.error_type == "mul":
                e_t = e_t/y_hat
        
            self.update_state(e_t)
            self.errors[ind] = e_t
    
    def predict(self, h, confidence=None):

        forecasts = torch.zeros(h)

        for step in range(1, h+1):
            damp_factor = torch.sum(DataLoader([self.phi**x for x in range(1, step+1)]).to_tensor())

            if self.seasonal is not None:
                seasonal_step = (step - 1) % self.seasonal_periods 

            y_hat = self.level.clone()

            if self.trend == "mul":
                y_hat *= self.trend_factor**damp_factor
            elif self.trend == "add":
                y_hat += self.trend_factor*damp_factor

            if self.seasonal == "mul":
                y_hat *= self.seasonal_factors[seasonal_step]
            elif self.seasonal == "add":
                y_hat += self.seasonal_factors[seasonal_step]

            forecasts[step-1] = y_hat

        if confidence is not None:
            if self.data_loader.is_valid_numeric(confidence) and (confidence < 1 and confidence > 0):
                upper_bounds, lower_bounds = self.calculate_confidence_interval(forecasts, float(confidence))

                return forecasts, (upper_bounds, lower_bounds)
        
        return forecasts