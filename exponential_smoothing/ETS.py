import numpy as np
import pandas as pd
import torch
from .model import ETS_Model
from exponential_smoothing.models.ets_models import *
from dataloader import DataLoader
import scipy.stats as stats

class ETS(ETS_Model):

    def __new__(self, dep_var, error_type="add", trend=None, damped=False, seasonal=None, seasonal_periods=None, indep_var=None, **kwargs):

        #keys are [trend, seasonal, error(add or mul)]
        ets_class = { 
                        (None, None, "add"): ETS_ANN,
                        (None, "add", "add"): ETS_ANA,
                        (None, "mul", "add"): ETS_ANM,
                        ("add",  None, "add"): ETS_AAN,
                        ("add", "add", "add"): ETS_AAA,
                        ("add", "mul", "add"): ETS_AAM,

                        (None,  None, "mul"): ETS_MNN,
                        (None,  "add", "mul"): ETS_MNA,
                        (None, "mul", "mul"): ETS_MNM,
                        ("add",  None, "mul"): ETS_MAN,
                        ("add", "add", "mul"): ETS_MAA,
                        ("add", "mul", "mul"): ETS_MAM,
                                                            }[trend, seasonal, error_type]

        return ets_class(dep_var, trend=trend, seasonal=seasonal, error_type=error_type, damped=damped, 
                         seasonal_periods=seasonal_periods, indep_var=None, **kwargs)
