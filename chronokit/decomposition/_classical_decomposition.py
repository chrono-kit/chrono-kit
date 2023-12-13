from chronokit.preprocessing._dataloader import DataLoader
from chronokit.utils.vis_utils import plot_decomp
import numpy as np
import pandas as pd


def classical_decomposition(data, seasonal_period, method="add", show=False):
    """
    Classical Decomposition for univariate time series data

    Arguments:

    *data (array_like): Time series data to perform decomposition on
    *seasonal_period (int): Seasonal period of the given data
    *method (Optional[str]): Decomposition method to be used; "add" or "mul"
    *show (Optional[bool]): Whether to plot the decomposition results

    Chapter 6.3 of the textbook is taken as a reference:
    'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice. OTexts, 2014.'
    """
    data = DataLoader(data).to_numpy()

    if data.ndim >= 1:
        data = np.squeeze(data)

    number_of_cycles = len(data) // seasonal_period
    assert number_of_cycles >= 2, "Data must have at least 2 full seasonal cycles"

    # Computing the trend-cycle component using moving averages
    trend = pd.Series(data).rolling(seasonal_period, center=True).mean()

    if seasonal_period % 2 == 0:
        trend = trend.shift(-1).rolling(2).mean()

    trend = trend.values

    # Detrending
    if method == "add":
        detrended = data - trend
    elif method == "mul":
        detrended = data / trend

    # Calculating the seasonal component
    seasonal = np.zeros(shape=(seasonal_period, number_of_cycles)) * np.nan
    for i in range(0, number_of_cycles):
        seasonal[:, i] = detrended[i * seasonal_period : (i + 1) * seasonal_period]
    seasonal = np.nanmean(seasonal, axis=1)

    # Putting seasonality component in a numpy array same length as the original data
    seasonal = np.array([seasonal[i % seasonal_period] for i in range(len(data))])

    # Deseasoning (Calculating remainder)
    if method == "add":
        remainder = detrended - seasonal
    elif method == "mul":
        remainder = detrended / seasonal

    if show:
        plot_decomp(trend, seasonal, remainder)

    return trend, seasonal, remainder
