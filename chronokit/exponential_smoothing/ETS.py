from .model import ETS_Model
from chronokit.exponential_smoothing.models.ets_models import *

class ETS(ETS_Model):

    def __new__(self, dep_var, error_type="add", trend=None, damped=False, seasonal=None, seasonal_periods=None, initialization_method="heuristic", **kwargs):
        """
        ETS (Error,Trend,Seasonality) model for time series data

        Arguments:

        *dep_var (array_like): Univariate time series data
        *error_type (str): Type of error of the ETS model; "add" or "mul"
        *trend (Optional[str]): Trend component; None or "add"
        *damped (bool): Damp factor of the trend component; False if trend is None
        *seasonal (Optional[str]): Seasonal component; None, "add" or "mul"
        *seasonal_periods (Optional[int]): Cyclic period of the seasonal component; int or None if seasonal is None
        *initialization_method (str): Initialization method to use for the model parameters; "heuristic" or "mle"

        Keyword Arguments:

        ** alpha (float): Smoothing parameter for level component; takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component; takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component; takes values in (0,1)
        
        ETS models are implemented by the below textbook as a reference:
        'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.'
        """
        
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
                         seasonal_periods=seasonal_periods, initialization_method=initialization_method, **kwargs)
