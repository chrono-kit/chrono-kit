import torch
import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.exponential_smoothing.__base_model import InnovationsStateSpace

"""
Exponential Smoothing Models for time series analysis and forecasting

EXponential Smoothing Models was developed with 
Hyndman, R. et al. [1] as a reference;

General Exponential Smoothing equation;
 
    y_t = w(x_{t-1})
    x_t = f(x_{t-1}) + g(y_t)

where x_t is the state vector at time t;
    x_t = (l_t, b_t, s0, s1, .., sm)

Note that unlike general state space models;
ExponentialSmoothing does not incorporate errors directly into the equations

However; these models are can be written in respective state space forms,
Since they still include y_t in state vector updates

With the same parameters and state vectors,
point forecasts for;

* ExponentialSmoothing(T, S)
* ETS(A, T, S)
* ETS(M, T, S)

should be equal, given that T = trend and S = seasonal

However, ETS models incorporate errors generated during fitting
into state vector updates; hence leads to different parameter 
estimations than Exponential Smoothing models.

Some ETS models also have algebraic expressions for probabilistic
forecast bounds given a confidence level. As they incorporate 
prediction errors directly into parameter estimations.

Exponential Smoothing models can only acquire forecast bounds through
simulating future sample paths

For further reading on ETS and Exponential Smoothing models;

Refer to;

[1] 'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
    Forecasting with exponential smoothing: The state space approach'

"""

class ExponentialSmoothing(InnovationsStateSpace):
    def __init__(
                self,
                data,
                trend=None,
                seasonal=None,
                seasonal_periods=None,
                damped=False,
                initialization_method="heuristic",
                **kwargs,
):
        """
        ExponentialSmoothing Models for 
        univariate time series modeling and forecasting

        Arguments:

        *data (array_like): Univariate time series data
        *trend (Optional[str]): Trend component; One of [None, "add", "mul]
        *seasonal (Optional[str]): Seasonal component; One of [None, "add", "mul]
            None if seasonal_periods is None
        *seasonal_periods (Optional[int]): Cyclic period of the
            seasonal component; None if seasonal is None
        *damped (bool): Damp factor of the trend component; One of [True, False]
            False if trend is None
        *initialization_method (Optional[str]): Initialization method to use
            for the model parameters; One of ["heuristic", "simple", "mle"]. 
            Default = 'heuristic'.

        Keyword Arguments:

        ** initial_level (float): Initial value for the level component;
        ** initial_trend_factor (float): Initial value for the trend component;
        ** initial_seasonal_factors (array_like): Initial values for seasonal components; 
            Must have length = seasonal_periods

        ** alpha (float): Smoothing parameter for level component;
            takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component;
            takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component;
            takes values in (0,1)

        Exponential Smoothing Models have been written with the below book as reference: 
            'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        """
        super().set_allowed_kwargs(["initial_level", 
                                    "initial_trend_factor",
                                    "initial_seasonal_factors",
                                    "alpha", "beta", "phi", "gamma"])

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        super().__init__(data, initialization_method, **kwargs)
        
    def update_state(self, y_t):
        """
        Update the state vector with model parameters and last error

        Equations are given in Hyndman, R. et al. Table 2.1

        Refer to Hyndman, R. et al. Chapter 2.5 for further information
        """
        lprev, bprev = self.level.clone(), self.trend_factor.clone()
        s_prev = self.seasonal_factors[0].clone()

        if self.seasonal == "mul":
            level_update_p1 = self.alpha*(y_t/s_prev)
            
            if self.trend == "mul":
                seasonal_update_p1 = self.gamma*y_t/(lprev*(bprev**self.phi))
                trend_update_p1 = (1-self.beta)*(bprev**self.phi)
                level_update_p2 = (1-self.alpha)*lprev*(bprev**self.phi)

            elif self.trend == "add":
                seasonal_update_p1 = self.gamma*y_t/(lprev + bprev*self.phi)
                trend_update_p1 = (1-self.beta)*(bprev*self.phi)
                level_update_p2 = (1-self.alpha)*(lprev + bprev*self.phi)
            
            else:
                seasonal_update_p1 = self.gamma*y_t/lprev
                level_update_p2 = (1-self.alpha)*lprev
        
        elif self.seasonal == "add":
            level_update_p1 = self.alpha*(y_t - s_prev)

            if self.trend == "mul":
                seasonal_update_p1 = self.gamma*(y_t - lprev*(bprev**self.phi))
                trend_update_p1 = (1-self.beta)*(bprev**self.phi)
                level_update_p2 = (1-self.alpha)*lprev*(bprev**self.phi)

            elif self.trend == "add":
                seasonal_update_p1 = self.gamma*(y_t - lprev - bprev*self.phi)
                trend_update_p1 = (1-self.beta)*(bprev*self.phi)
                level_update_p2 = (1-self.alpha)*(lprev + bprev*self.phi)
            
            else:
                seasonal_update_p1 = self.gamma*(y_t - lprev)
                level_update_p2 = (1-self.alpha)*lprev
        
        else:
            level_update_p1 = self.alpha*y_t
            
            if self.trend == "mul":
                trend_update_p1 = (1-self.beta)*(bprev**self.phi)
                level_update_p2 = (1-self.alpha)*lprev*(bprev**self.phi)

            elif self.trend == "add":
                trend_update_p1 = (1-self.beta)*(bprev*self.phi)
                level_update_p2 = (1-self.alpha)*(lprev + bprev*self.phi)
            
            else:
                level_update_p2 = (1-self.alpha)*lprev
         
        self.level = level_update_p1 + level_update_p2

        if self.trend == "mul":
            self.trend_factor = self.beta*(self.level/lprev) + trend_update_p1
        elif self.trend == "add":
            self.trend_factor = self.beta*(self.level - lprev) + trend_update_p1

        if self.seasonal is not None:
            seasonal = seasonal_update_p1 + (1-self.gamma)*s_prev
            self.seasonal_factors = torch.cat((self.seasonal_factors[1:], torch.tensor([seasonal])), dim=-1)
    
    def fit(self):
        """Fit the model to the given data"""

        #Ensure valid parameters
        self.seasonal_factors = DataLoader(self.seasonal_factors).to_tensor()
        self.level = DataLoader(self.level).to_tensor()
        self.trend_factor = DataLoader(self.trend_factor).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        for ind, y in enumerate(self.data):
            if ind == 0:
                self.fitted[0] = torch.nan
                continue
            
            y_hat = self.predict(1)
            self.fitted[ind] = y_hat
            self.update_state(y)
    
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