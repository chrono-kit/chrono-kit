import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from exponential_smoothing import ETS
import torch

def mle_initialization(data, trend=None, seasonal=None, error_type="add", damped=False, seasonal_periods=None):
    def log_likelihood(parameters, data):
        alpha = parameters[0]
        n = len(data)
        # Initial level
        l0 = parameters[1]

        # Initial trend
        b0 = (data[-1] - data[0]) / (n - 1)


        # Initialize log-likelihood
        log_likelihood_value = 0.0

        # Calculate log-likelihood
        for i in range(n):
            if i == 0:
                level = l0
                trend = b0
            else:
                level = level_1 + error
                trend = trend_1 + alpha * error

            error = data[i] - level - trend
            log_likelihood_value += norm.logpdf(error, loc=0, scale=1)

            level_1 = level
            trend_1 = trend

        return -log_likelihood_value

    # Use the optimization function to find the maximum likelihood estimates with random initial values
    result_params = opt.minimize(log_likelihood,[0.3,data.mean()], args=(data,), method='L-BFGS-B')

    return result_params.x

data = pd.read_csv('datasets/AirPassengers.csv')
initial_params = mle_initialization(data['#Passengers'].values, trend="add", damped=True, seasonal="mul", seasonal_periods=8)
print(initial_params)
