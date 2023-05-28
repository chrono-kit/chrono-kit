import numpy as np
import pandas as pd
import torch

def heuristic_initialization(data, trend=False, seasonal=False,
                              seasonal_periods=None):
    # See Section 2.6 of Hyndman et al.
    data = torch.clone(data).numpy()

    if data.ndim >= 1:

        data = np.squeeze(data)

    data_len = len(data)

    assert(data_len >= 10), 'Cannot use heuristic method with less than 10 observations.'

    # Seasonal component
    initial_seasonal = None
    if seasonal:
        # Calculate the number of full cycles to use
        assert(data_len >= 2 * seasonal_periods), 'Cannot compute initial seasonals using heuristic method with less than two full \
                                                    seasonal cycles in the data.'
        
        # We need at least 10 periods for the level initialization
        # and we will lose self.seasonal_periods // 2 values at the
        # beginning and end of the sample, so we need at least
        # 10 + 2 * (self.seasonal_periods // 2) values

        min_len = 10 + 2 * (seasonal_periods // 2)
        assert(data_len >= min_len), 'Cannot use heuristic method to compute \
                             initial seasonal and levels with less \
                             than 10 + 2 * (seasonal_periods // 2) \
                             datapoints.'
        
        # In some datasets we may only have 2 full cycles (but this may
        # still satisfy the above restriction that we will end up with
        # 10 seasonally adjusted observations)
        k_cycles = min(5, data_len // seasonal_periods)

        # In other datasets, 3 full cycles may not be enough to end up
        # with 10 seasonally adjusted observations

        k_cycles = max(k_cycles, int(np.ceil(min_len / seasonal_periods)))

        # Compute the moving average
        series = pd.Series(data[:seasonal_periods * k_cycles])
        initial_trend = series.rolling(seasonal_periods, center=True).mean()

        if seasonal_periods % 2 == 0:
            initial_trend = initial_trend.shift(-1).rolling(2).mean()

        # Detrend
        if seasonal == 'add':
            detrended = series - initial_trend
        elif seasonal == 'mul':
            detrended = series / initial_trend

        # Average seasonal effect
        tmp = np.zeros(k_cycles * seasonal_periods) * np.nan
        tmp[:len(detrended)] = detrended.values
        initial_seasonal = np.nanmean(
            tmp.reshape(k_cycles, seasonal_periods).T, axis=1)

        # Normalize the seasonals
        if seasonal == 'add':
            initial_seasonal -= np.mean(initial_seasonal)
        elif seasonal == 'mul':
            initial_seasonal /= np.mean(initial_seasonal)

        # Replace the data with the trend
        data = initial_trend.dropna().values

    # Trend / Level
    exog = np.c_[np.ones(10), np.arange(10) + 1]
    if data.ndim == 1:
        data = np.atleast_2d(data).T
    beta = np.squeeze(np.linalg.pinv(exog).dot(data[:10]))
    initial_level = beta[0]

    initial_trend = None
    if trend == 'add':
        initial_trend = beta[1]
    elif trend == 'mul':
        initial_trend = 1 + beta[1] / beta[0]

    return initial_level, initial_trend, initial_seasonal