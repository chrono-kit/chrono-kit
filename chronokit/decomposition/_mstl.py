import numpy as np
import pandas as pd
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.utils.vis_utils import plot_decomp
from chronokit.decomposition._stl import STL

def MSTL(
    data,
    seasonal_periods,
    method="add",
    degrees=None,
    robust=True,
    refine_iterations=1,
    outer_iterations=10,
    inner_iterations=2,
    post_smoothing=False,
    show=False,
    **kwargs,
):
    """
    Seasonal Trend Decomposition using LOESS for univariate time series data.

    Arguments:

    *data (array_like): Univariate time series data to perform decomposition on.
    *seasonal_periods (array_like): Seasonality periods of each seasonality component.
        Must be ordered from the most granular seasonality to the least granular.
    *method Optional[str]: Decomposition method to be used; "add" or "mul"
    *degrees Optional[array_like]: Degrees to be used when performing STL for each of the seasonal
        component.Usually degree=0 or degree=1.
    *robust Optional[bool]: Whether to use robustness weights for LOESS smoothing. Default = True.
    *refine_iterations Optional[int]: Number of iterations when refining the seasonal components.
        Default=1.
    *outer_iterations Optional[int]: Number of iterations in the outer loop of each STL operaton.
        Default = 10.
    *inner_iterations Optional[int]: Number of iterations in the inner loop of each STL operation.
        Defaut = 2.
    *post_smoothing Optional[bool]: Whether to perform a final LOESS smoothing for trend and
        seasonal components for each STL. Default = False.
    *show Optional[bool]: Whether to plot the decomposition results. Default = False.

    Keyword Arguments:

    **trend_windows (array_like): Window size to use on each STL for the trend component.
    **seasonal_windows (array_like): Window size to use on each STL for the seasonal components.
    **low_pass_windows (array_like): Window size to use on each STL for low pass filtering.
    **trend_degrees (array_like): Degree used on each STL for the trend component.
        Will override the argument *degree for the trend component.
    **seasonal_degrees (array_like): Degree used on each STL for the seasonal component.
        Will override the argument *degree for the seasonal component.
    **low_pass_degrees (array_like): Degree used on each STL for the low pass filtering.
        Will override the argument *degree for the low pass filtering.

    Returns:

    *trend (np.ndarray): Trend component of the data
    *seasonals (np.ndarray): Seasonal component of the data
    *remainder (np.ndarray): Remainder component of the data
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

    # Assertions for arguments and keyword arguments
    try:
        for period in seasonal_periods:
            assert period >= 2, "Each seasonal_period must be >= 2"
            assert (
                len(data) > 2 * period
            ), "Data must have at least 2 full seasonal cycles for each seasonal period"

        seasonal_periods = list(seasonal_periods)
        assert len(seasonal_periods) >= 1, "Length of seasonal_periods must be >= 1"
    except:  # noqa: E722
        raise TypeError("seasonal_periods must be an iterable")

    if degrees is not None:
        try:
            iter(degrees)
            degrees = list(degrees)
            if len(degrees) != len(seasonal_periods):
                if len(degrees) > len(seasonal_periods):
                    degrees = degrees[: len(seasonal_periods)]
                else:
                    degrees.extend(seasonal_periods[len(degrees) :])
        except:  # noqa: E722
            raise TypeError("degrees must be an iterable")

    # Take degrees as 1 for all seasonal_periods
    else:
        degrees = [1 for i in seasonal_periods]

    allowed_kwargs = [
        "trend_windows",
        "seasonal_windows",
        "low_pass_windows",
        "trend_degrees",
        "seasonal_degrees",
        "low_pass_degrees",
    ]

    # Get the keyword arguments
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise ValueError("{key} is not a valid keyword for this model".format(key=k))
        locals()[k] = v

    trend_windows = locals().get("trend_window", [None for i in seasonal_periods])
    seasonal_windows = locals().get("seasonal_window", [None for i in seasonal_periods])
    low_pass_windows = locals().get("low_pass_window", [None for i in seasonal_periods])
    trend_degrees = locals().get("trend_degree", degrees)
    seasonal_degrees = locals().get("seasonal_degree", degrees)
    low_pass_degrees = locals().get("low_pass_degree", degrees)

    # Take the logarithm if multiplicative decomposition
    # y=trend*seasonals*remainder -> ln(y)=ln(trend)+ln(seasonal)+ln(remainder)
    if method == "mul":
        assert data.min() > 0, "Data must be strictly positive for multiplicative decomposition"
        data = np.log(data)

    # Initialize trend as 0
    trend = np.zeros(len(data))

    # Define an empty array of shape [num_seasonals, num_observations] to store seasonal components
    seasonals = np.zeros(shape=(len(seasonal_periods), len(data)))

    # Loop over the seasonal periods
    for ind, period in enumerate(seasonal_periods):
        # Run STL starting from the most granular seasonality and get the seasonal component
        _, s, _ = STL(
            data=data,
            seasonal_period=period,
            method="add",
            degree=degrees[ind],
            robust=robust,
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
            post_smoothing=post_smoothing,
            show=False,
            trend_window=trend_windows[ind],
            seasonal_window=seasonal_windows[ind],
            low_pass_window=low_pass_windows[ind],
            trend_degree=trend_degrees[ind],
            seasonal_degree=seasonal_degrees[ind],
            low_pass_degree=low_pass_degrees[ind],
        )

        # Deseasonalize the data to remove current seasonal component's effect
        data -= s

        # Store the current seasonal component inside seasonals array
        seasonals[ind, :] = s

    for _ in range(refine_iterations):
        # Loop over the seasonal periods again to refine the seasonal components
        for ind, period in enumerate(seasonal_periods):
            # Get the current seasonal component
            s = seasonals[ind, :]

            # Add the seasonality back to the data for refining
            data += s

            # Extract the trend and refined seasonal component
            # The trend component of the decomposition will be the final trend extracted
            # from refining the least granular seasonality
            trend, s, _ = STL(
                data=data,
                seasonal_period=period,
                method="add",
                degree=degrees[ind],
                robust=robust,
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                post_smoothing=post_smoothing,
                show=False,
                trend_window=trend_windows[ind],
                seasonal_window=seasonal_windows[ind],
                low_pass_window=low_pass_windows[ind],
                trend_degree=trend_degrees[ind],
                seasonal_degree=seasonal_degrees[ind],
                low_pass_degree=low_pass_degrees[ind],
            )

            # Store the refined seasonal component in the seasonal array
            seasonals[ind, :] = s

            # Deseasonalize the data again, this time using the refined seasonality
            data -= s

    # After refining, the data is deseasonalized,i.e;
    # data = y - seasonal_1 - seasonal_2 - ... - seasonal_n
    # Therefore, extract the remainder by removing trend component from data
    remainder = data - trend

    # Take the exponents to get actual values for multiplicative decomposition
    if method == "mul":
        trend = np.exp(trend)
        seasonals = np.exp(seasonals)
        remainder = np.exp(remainder)

    # Plot the results
    if show:
        plot_decomp(trend, seasonals, remainder, method=method)

    return trend, seasonals, remainder
