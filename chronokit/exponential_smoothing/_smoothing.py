from chronokit.base._smoothing_models import Smoothing_Model
from chronokit.exponential_smoothing.models.smoothing import SES, HoltTrend, HoltWinters


class ExponentialSmoothing:
    def __new__(
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
        Exponential Smoothing model for time series data

        Arguments:

        *data (array_like): Univariate time series data
        *trend (Optional[str]): Trend component; None or "add"
        *seasonal (Optional[str]): Seasonal component; None, "add" or "mul"
        *seasonal_periods (Optional[int]): Cyclic period of the seasonal component;
            int or None if seasonal is None
        *damped (Optional[bool]): Damp factor of the trend component; False if trend is None
        *initialization_method (str): Initialization method to use for the model parameters;
            "heuristic" or "mle"

        Keyword Arguments:

        ** alpha (float): Smoothing parameter for level component; takes values in (0,1)
        ** beta (float): Smoothing parameter for trend component; takes values in (0,1)
        ** phi (float): Damp factor for trend component; takes values in (0,1]
        ** gamma (float): Smoothing parameter for seasonal component; takes values in (0,1)

        ETS models are implemented by the below textbook as a reference:
        'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.'
        """

        smoothing_class = {
            (None, None): SES,
            ("add", None): HoltTrend,
            ("add", "add"): HoltWinters,
            ("add", "mul"): HoltWinters,
        }[trend, seasonal]

        return smoothing_class(
            data,
            trend=trend,
            seasonal=seasonal,
            damped=damped,
            seasonal_periods=seasonal_periods,
            initialization_method=initialization_method,
            **kwargs,
        )
