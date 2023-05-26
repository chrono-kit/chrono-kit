from .model import Smoothing_Model
from exponential_smoothing.models.smoothing import *


class ExponentialSmoothing:

    def __new__(self, dep_var, trend=None, damped=False, seasonal=None, seasonal_periods=None, indep_var=None, **kwargs):

    
        #keys are [trend, seasonal]
        smoothing_class = { 
                        (None, None): SES,
                        ("add", None): HoltTrend,
                        ("add", "add"): HoltWinters,
                        ("add", "mul"): HoltWinters,
                                                    }[trend, seasonal]

        return smoothing_class(dep_var, trend=trend, seasonal=seasonal, damped=damped, seasonal_periods=seasonal_periods, indep_var=indep_var, **kwargs)
