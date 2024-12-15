import numpy as np
import pandas as pd
from chronokit.preprocessing._dataloader import DataLoader
from scipy.linalg import lstsq
from scipy.interpolate import interp1d
from chronokit.utils.vis_utils import plot_decomp

def bi_square(x):
    """
    The bi-square function for robust regression in LOWESS and LOESS.

    Arguments:

    *x (array_like): Input to pass through the bi-square function.

    Returns:

    *B(x) (array_like): (1-x^2)^2 if |x| < 1 else 0.
    """

    # Convert to numpy array if x is not a numpy array
    if not isinstance(x, np.ndarray):
        x = DataLoader(x).to_numpy()

    # The bi-square function
    return np.where(np.abs(x) < 1, (1 - abs(x) ** 2) ** 2, 0)


def tri_cube(x):
    """
    The tri-cube function that is used as a weight function for fitted points in LOWESS and LOESS.

    *Arguments:

    *x (array_like): Input to pass through the bi-square function.

    Returns:

    *w(x) (array_like): (1-|x|^3)^3 if |x| < 1 else 0.
    """

    # Convert to numpy array if x is not a numpy array
    if not isinstance(x, np.ndarray):
        x = DataLoader(x).to_numpy()

    # The tri-cube function
    return np.where(np.abs(x) < 1, (1 - abs(x) ** 3) ** 3, 0)


def LOESS(data, window_size, degree=1, robustness_weights=None):
    #return lowess(data, exog=range(len(data)), it=1, frac=window_size/len(data))[:, 1]
    """
    Locally Estimated Scatterplot Smoothing for time series data

    Arguments:

    *data (array_like): Univariate time series data, must be data.ndim == 1
    *window_size (int): Window size for weighting data points.
    *degree Optional[int]: Degree used for fitting polynomial p[x] = y. Usually degree=0 or degree=1
    *robustness_weights Optional[array_like]: Weights used for robustness to outliers.
    Defaults to None

    Returns:

    *y_smoothed (np.ndarray): Smoothed values as a result of LOESS.
    """
    
    # Define a loess function fitting to all of the data at once
    def loess(x, y, xi, degree=1, robustness_weights=None, q=None):
        """
        *x (np.ndarray): Data indexes to fit
        *y (np.ndarray): Data to fit, must be ndim == 1
        *xi (int): Index to assign weight == 1. I.e;
            the point where the smoothing result matters the most
        *degree (int): Degree of the polynomial p[x] = y
        *robustness_weights (np.ndarray): Robustness weights for outliers.
        *q (int): Distance for weighting data points.
            All x s.t distance(x-xi) >= q will be assigned weights=0. Must be odd.
        """

        if q:
            # If q is greater than the max distance from xi, take q as max_dist*q/len(y)
            max_dist = max(abs(x[0] - xi), abs(x[-1] - xi))
            if q > max_dist:
                q = int(max_dist * q / len(y))

        # Take q as the half the length of the data to be fit if not given.
        elif not q:
            q = int((len(y) - 1) / 2)

        assert (isinstance(degree, int) and degree >= 0), "Degree must be a non-negative integer"
        
        if xi - q > 0 and xi + q < len(y):
            subset_start = max(0, xi - q)
            subset_end = min(len(y), xi + q+1)
        elif xi - q <=0:
            subset_start = 0
            subset_end = q*2 + 1
        elif xi + q >= len(y):
            subset_end = len(y)
            subset_start = len(y) - q*2 - 1

        distances = np.abs(x - xi)
        distances = distances[subset_start:subset_end]
            
        x = x[subset_start:subset_end]
        y = y[subset_start:subset_end]

        # Calculate distances of each data point to xi
        # Calculate weights by the tricube function, making sure that data points
        # farther than q are given weight 0
        weights = tri_cube(distances / np.max(distances))
        
        # If robustness weights are given, multiply weights so that outliers are given less weights
        if robustness_weights is not None:
            weights *= robustness_weights[subset_start:subset_end]
        
        #Normalize weights
        weights /= weights.sum()

        # Get the vandermonde matrix of x for polynomial fitting
        A = np.vander(x, degree+1)

        # Get the w from the fitted equation A*w = y
        # Multiplying A and y by sqrt(weights) to perform weighted least squares
        w, _, _, _ = lstsq(A * np.sqrt(weights[:, np.newaxis]), y * np.sqrt(weights))

        return np.polyval(w, xi)

    # Calculate the half of the window size to use for weighting data points in loess(used as q),
    half_window = int((window_size - 1) / 2)

    # Turn data into np.ndarray if it is not
    if not isinstance(data, np.ndarray):
        data = DataLoader(data).to_numpy()

    # Squeeze if ndim > 1
    if data.ndim > 1:
        shape = data.shape

        if shape[0] == 1:
            data = data.squeeze(0)
        elif shape[1] == 1:
            data = data.squeeze(1)
        # If cannot squeeze to ndim==1, raise ValueError
        else:
            raise ValueError("data.ndim must be == 1 or squeezable to ndim==1")
        if data.ndim > 1:
            raise ValueError("data.ndim must be == 1 or squeezable to ndim==1")

    # Define empty array to store LOESS results
    smoothed = np.empty(len(data))

    # Define an array indexing each data point
    x = np.arange(len(data))

    # For each data point(xi), fit loess by giving full weight to xi and
    # store the result in the smoothed array
    
    for xi in x:
        # Fit loess by weights centered around xi
        # Store the fitted value for data[xi] in the smoothed array
        smoothed[xi] = loess(
            x,
            data,
            xi,
            degree=degree,
            robustness_weights=robustness_weights,
            q=half_window,
        )

    return smoothed


def __inner_loop(
    y,
    trend,
    weights,
    seasonal_period,
    seasonal_degree=1,
    trend_degree=1,
    low_pass_degree=1,
    seasonal_smoothing=None,
    trend_smoothing=None,
    low_pass_smoothing=None,
    n_iterations=2,
):
    """
    Inner loop of the Seasoal-Trend Decomposition using LOESS (STL).

    *y (np.ndarray): Univariate time series data
    *trend (np.ndarray): Trend component of the data, is 0 for the first iteration
    *weights (np.ndarray): Robustness weights returned from outer loop.
    *seasonal_period (int): Seasonality period of the data
    *seasonal_degree Optional[int]: Degree to use on LOESS for seasonal component.
    *trend_degree Optional[int]: Degree to use on LOESS for trend component.
    *low_pass_degree Optional[int]: Degree to use on LOESS for low pass filtering.
    *seasonal_smoothing int: Window size to use on LOESS for the seasonal component
    *trend_smoothing int: Window size to use on LOESS for the trend component.
    *low_pass_smoothing int: Window size to use on LOESS for low pass filtering.
    *n_iterations Optional[int]: Number of iterations for the inner loop.
    """

    for i in range(n_iterations):
        # Define empty array for storing smoothing results for cycle_subseries
        ct = np.zeros(len(y) + 2 * seasonal_period)

        # Detrend by subtracting trend component
        detrended = y - trend
        
        for i in range(seasonal_period):
            # Get each cycle-subseries from detrended data
            subseries = detrended[range(i, len(y), seasonal_period)]

            # Get the robustness weights corresponding to current cycle-subseries
            s_weights = weights[range(i, len(weights), seasonal_period)]
            
            # Get the results of the LOESS smoothing for the current cycle-subseries
            loess_res = LOESS(
                data=subseries,
                window_size=seasonal_smoothing,
                degree=seasonal_degree,
                robustness_weights=s_weights,
            )

            # For the 2*seasonal_periods amount of values that will be lost during low pass filtering;
            # Extrapolate and extend the results by 1 time step amount
            extrapolate = interp1d(np.arange(len(loess_res)), loess_res, fill_value="extrapolate")
            loess_res = [extrapolate(-1)] + list(loess_res) + [extrapolate(len(loess_res))]
            # Store the results of the LOESS on the current cycle-subseris in the ct array.
            ct[range(i, len(ct), seasonal_period)] = np.array(loess_res)

        # Define a pd.Series equal to ct for low pass fitering
        lt = pd.Series(np.copy(ct))

        # Perform two centered moving averages of length seasonal_period
        if seasonal_period % 2 == 0:
            # If the seasonal_period is even, ensure the centered moving averages result in,
            # seasonal_period amount of data points are NaN at each end point
            lt = lt.rolling(seasonal_period).mean()
            lt = lt[-1::-1].rolling(seasonal_period).mean()
            lt = lt[-1::-1]
        else:
            # If the seasonal_period is odd, perform centered moving averages normally
            lt = lt.rolling(seasonal_period, center=True).mean()
            lt = lt.rolling(seasonal_period, center=True).mean()

        # Perform another centered moving average of length 3
        lt = lt.rolling(3, center=True).mean()
        # Drop the NaN values
        lt = lt.iloc[seasonal_period:-seasonal_period].values
        # Perform LOESS on the low pass filter
        lt = LOESS(
            data=lt,
            window_size=low_pass_smoothing,
            degree=low_pass_degree,
            robustness_weights=None,
        )
        # Calculate seasonal component by ct-lt
        seasonal = ct[seasonal_period:-seasonal_period] - lt

        # Calculate the trend component by de-seasonalizing
        trend = y - seasonal
        # Perform LOESS on the trend component
        trend = LOESS(
            data=trend,
            window_size=trend_smoothing,
            degree=trend_degree,
            robustness_weights=weights,
        )

    return trend, seasonal


def __outer_loop(y, trend, seasonal):
    """
    Outer loop of the Seasoal-Trend Decomposition using LOESS (STL).

    *y (np.ndarray): Univariate time series data
    *trend (np.ndarray): Trend component of the data
    *seasonal (np.ndarray): Seasonal component of the data
    """

    # Calculate the remainder
    remainder = y - trend - seasonal

    # Calculate robustness weights based on the remainder
    # Ensure that any remainder >= h is assigned a weight of 0
    # The bigger the remainder, the less weight given to the datapoint
    h = 6 * np.median(abs(remainder))
    weights = bi_square(abs(remainder) / h)

    return weights, remainder


def STL(
    data,
    seasonal_period,
    method="add",
    degree=1,
    robust=True,
    outer_iterations=None,
    inner_iterations=None,
    post_smoothing=False,
    show=False,
    **kwargs,
):
    """
    Seasonal Trend Decomposition using LOESS for univariate time series data.

    Arguments:

    *data (array_like): Univariate time series data to perform decomposition on.
    *seasonal_period (int): Seasonality period of the data.
    *method Optional[str]: Decomposition method to be used; "add" or "mul"
    *degree Optional[int]: Degree used for fitting polynomial p[x] = y on LOESS.
        Usually degree=0 or degree=1.
    *robust Optional[bool]: Whether to use robustness weights for LOESS smoothing. Default = True.
    *outer_iterations Optional[int]: Number of iterations in the outer loop. Default = 10.
    *inner_iterations Optional[int]: Number of iterations in the inner loop. Defaut = 2.
    *post_smoothing Optional[bool]: Whether to perform a final LOESS smoothing for trend and
        seasonal components post operation. Default = False.
    *show Optional[bool]: Whether to plot the decomposition results. Default = False.

    Keyword Arguments:

    **trend_window (int): Window size to use on LOESS smoothing for the trend component.
    **seasonal_window (int): Window size to use on LOESS smoothing for the seasonal component.
    **low_pass_window (int): Window size to use on LOESS smoothing for low pass filtering.
    **trend_degree (int): Degree used on LOESS for the trend component.
        Will override the argument *degree for the trend component.
    **seasonal_degree (int): Degree used on LOESS for the seasonal component.
        Will override the argument *degree for the seasonal component.
    **low_pass_degree (int): Degree used on LOESS for the low pass filtering.
        Will override the argument *degree for the low pass filtering.

    Returns:

    *trend (np.ndarray): Trend component of the data
    *seasonal (np.ndarray): Seasonal component of the data
    *remainder (np.ndarray): Remainder component of the data

    References:
    https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf
    """

    assert method == "add" or method == "mul", "Only method='add' or method='mul' is supported"

    # Turn data into np.ndarray if it is not
    data = DataLoader(data).to_numpy()

    # Squeeze if ndim > 1
    if data.ndim > 1:
        shape = data.shape

        if shape[0] == 1:
            data = data.squeeze(0)
        elif shape[1] == 1:
            data = data.squeeze(1)
        # If cannot squeeze to ndim==1, raise ValueError
        else:
            raise ValueError("data.ndim must be == 1 or squeezable to ndim==1")
        if data.ndim > 1:
            raise ValueError("data.ndim must be == 1 or squeezable to ndim==1")

    # Assertions for making sure parameters are valid
    assert seasonal_period >= 2, "seasonal_period must be >= 2"
    assert len(data) > 2 * seasonal_period, "Data must have at least 2 full seasonal cycles"

    allowed_kwargs = [
        "trend_window",
        "seasonal_window",
        "low_pass_window",
        "trend_degree",
        "seasonal_degree",
        "low_pass_degree",
    ]

    # Get the keyword arguments
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise ValueError("{key} is not a valid keyword for this model".format(key=k))

    trend_window = kwargs.get("trend_window", None)
    seasonal_window = kwargs.get("seasonal_window", None)
    low_pass_window = kwargs.get("low_pass_window", None)
    trend_degree = kwargs.get("trend_degree", degree)
    seasonal_degree = kwargs.get("seasonal_degree", degree)
    low_pass_degree = kwargs.get("low_pass_degree", degree)

    # Assertions for making sure parameters are valid
    assert degree >= 0, "Degree must be greater than 0"
    assert trend_degree >= 0, "Trend degree must be greater than 0"
    assert seasonal_degree >= 0, "Seasonal degree must be greater than 0"
    assert low_pass_degree >= 0, "Low pass degree must be greater than 0"

    # If low pass window is not given,take it as the least odd integer greater than or
    # equal to seasonal period
    if not low_pass_window:
        if seasonal_period % 2 == 1:
            low_pass_window = seasonal_period
        else:
            low_pass_window = seasonal_period + 1

    # If seasonal_window is not given, take it as 7
    if not seasonal_window:
        seasonal_window = 7

    # If trend_window is not given, take it as the least odd integer,denote it as k, satisfying;
    # k >= 1.5*seasonal_period/(1- 1/(1.5*seasonal_window))
    if not trend_window:
        trend_window = 1.5 * seasonal_period / (1 - 1.5/seasonal_window)
        if trend_window != int(trend_window):
            trend_window = int(trend_window) + 1
        else:
            trend_window = int(trend_window)
        if trend_window % 2 == 0:
            trend_window += 1
    
    if outer_iterations is None:
        outer_iterations = 15 if robust else 0
    if inner_iterations is None:
        inner_iterations = 2 if robust else 5

    # Assertions for making sure parameters are valid
    assert outer_iterations >= 0, "Number of outer loop iterations must be >= 0"
    assert inner_iterations >= 1, "Number of inner loop iterations must be >= 1"
    assert (
        low_pass_window % 2 == 1 and low_pass_window >= seasonal_period
    ), "Window size for low pass filtering must be odd and greater than or equal to \
        the seasonal period"
    assert (
        seasonal_window % 2 == 1 and seasonal_window >= 7
    ), "Window size for seasonal smoothing must be odd and greater than or equal to 7"
    assert (
        trend_window % 2 == 1 and trend_window > 1
    ), "Window size for trend smoothing must be odd and greater than 1"

    # Take the logarithm if multiplicative decomposition
    # y=trend*seasonal*remainder -> ln(y)=ln(trend)+ln(seasonal)+ln(remainder)
    if method == "mul":
        assert data.min() > 0, "Data must be strictly positive for multiplicative decomposition"
        data = np.log(data)

    # Initialize trend as 0
    trend = np.zeros(len(data))

    # Initialize robustness weight as 1
    weights = np.ones(len(data))

    for _ in range(outer_iterations+1):
        # Perform the inner loop and get the trend and seasonal components
        trend, seasonal = __inner_loop(
            y=data,
            trend=trend,
            weights=weights,
            seasonal_period=seasonal_period,
            seasonal_degree=seasonal_degree,
            trend_degree=trend_degree,
            low_pass_degree=low_pass_degree,
            seasonal_smoothing=seasonal_window,
            trend_smoothing=trend_window,
            low_pass_smoothing=low_pass_window,
            n_iterations=inner_iterations,
        )

        # Perform the outer loop and get the remainder and robustness weights
        weights, remainder = __outer_loop(y=data, trend=trend, seasonal=seasonal)

        # If not robust, keep robustness weights as 1.
        if not robust:
            weights = np.ones(len(data))

    # If post smoothing, perform one last LOESS on the components
    if post_smoothing:
        seasonal = LOESS(data=seasonal, window_size=seasonal_window, degree=seasonal_degree)
        trend = data - seasonal
        trend = LOESS(data=trend, window_size=trend_window, degree=trend_degree)
        remainder = data - seasonal - trend

    # Take the exponents to get actual values for multiplicative decomposition
    if method == "mul":
        trend = np.exp(trend)
        seasonal = np.exp(seasonal)
        remainder = np.exp(remainder)

    # Plot the results:
    if show:
        plot_decomp(trend, seasonal, remainder, method=method)

    return (
        DataLoader(trend).to_numpy(),
        DataLoader(seasonal).to_numpy(),
        DataLoader(remainder).to_numpy(),
    )