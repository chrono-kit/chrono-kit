import torch
import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.base._models import Model
from chronokit.exponential_smoothing._initialization import SmoothingInitializer

class ExponentialSmoothing(Model):
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
        Base class for exponential smoothing methods.
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
        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(data, **kwargs)

        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        if self.seasonal == "mul":
            assert (
                DataLoader(data).to_numpy().min() > 0
            ), "Methods with multiplicative seasonality requires \
                data to be strictly positive"
        
        self.initializer = SmoothingInitializer(self, method=initialization_method)
        self.initializer.initialize_parameters()

        self.params = self.initializer.init_params.copy()

        self.alpha = self.params["alpha"]
        self.beta = self.params["beta"]
        self.gamma = self.params["gamma"]
        self.phi = self.params["phi"]

        self.init_components = self.initializer.init_components.copy()
        self.components = self.initializer.init_components.copy()

        self.level = self.components["level"]
        self.trend_factor = self.components["trend_factor"]
        self.seasonal_factors = self.components["seasonal_factors"]

    def update_params(self, y_t):

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
            self.update_params(y)
    
    def predict(self, h):

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
        
        if h == 1:
            return forecasts[0]

        else:
            return forecasts