from .model import Smoothing_Model
from exponential_smoothing.models.smoothing import *


class ExponentialSmoothing:

    def __new__(self, dep_var, trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, indep_var=None, **kwargs):

        self.trend = trend
        self.damped = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        #keys are [trend, seasonal]
        smoothing_class = { 
                        (None, None): SES,
                        ("add", None): HoltTrend,
                        ("add", "add"): HoltWinters,
                        ("add", "mul"): HoltWinters,
                                                    }[self.trend, self.seasonal]

        return smoothing_class(dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods, 
                               damped=self.damped, indep_var=None, **kwargs)
