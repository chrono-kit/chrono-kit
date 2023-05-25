from .model import Smoothing_Model
from exponential_smoothing.models.smoothing import *


class ExponentialSmoothing:

    def __new__(self, dep_var, trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, **kwargs):

        #keys are [trend, seasonal]
        smoothing_class = { 
                        (None, None): SES,
                        ("add", None): HoltTrend,
                        ("add", "add"): HoltWinters,
                        ("add", "mul"): HoltWinters,
                                                    }[trend, seasonal]

        return smoothing_class(dep_var, damped=damped_trend, seasonal=seasonal, seasonal_periods=seasonal_periods, **kwargs)
