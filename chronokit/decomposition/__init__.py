from ._classical_decomposition import classical_decomposition
from ._stl import STL, LOESS
from ._mstl import MSTL


def LOWESS(data, window_size):
    """
    Locally Weighted Estimated Scatterplot Smoothing for time series data

    Arguments:

    *data (array_like): Univariate time seriessdata, must be data.ndim == 1
    *window_size (int): Window size for weighting data points.

    Returns:

    *smootheed (np.ndarray): Smoothed values as a result of LOWESS.
    """

    return LOESS(data, window_size, degree=1, robustness_weights=None)
