import numpy as np
import pandas as pd
import torch
import scipy.optimize as opt
from scipy.stats import norm
from exponential_smoothing.initialization import ets_methods

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

def mle_initialization(data, trend=None,damped=False, seasonal=None, error_type="add",seasonal_periods=12,alpha=0,beta=0,gamma=0,phi=0):
    # Select the error function of the model determined by trend, damped, seasonal and error_type parameters.
    model_err = {
        (None, False, None, "add"): ets_methods.ANN,
        (None, False, "add", "add"): ets_methods.ANA,
        (None, False, "mul", "add"): ets_methods.ANM,
        ("add", False, None, "add"): ets_methods.AAN,
        ("add", False, "add", "add"): ets_methods.AAA,
        ("add", False, "mul", "add"): ets_methods.AAM,
        ("add", True, None, "add"): ets_methods.AAdN,
        ("add", True, "add", "add"): ets_methods.AAdA,
        ("add", True, "mul", "add"): ets_methods.AAdM,
        (None, False, None, "mul"): ets_methods.MNN,
        (None, False, "add", "mul"): ets_methods.MNA,
        (None, False, "mul", "mul"): ets_methods.MNM,
        ("add", False, None, "mul"): ets_methods.MAN,
        ("add", False, "add", "mul"): ets_methods.MAA,
        ("add", False, "mul", "mul"): ets_methods.MAM,
        ("add", True, None, "mul"): ets_methods.MAdN,
        ("add", True, "add", "mul"): ets_methods.MAdA,
        ("add", True, "mul", "mul"): ets_methods.MAdM
        }
    selected_model = model_err[(trend, damped, seasonal, error_type)]

    # Find the parameters of selected model to maximize the likelihood function.
    params = {
        (None, False, None, "add"): [alpha],
        (None, False, "add", "add"): [alpha, gamma],
        (None, False, "mul", "add"): [alpha, gamma],
        ("add", False, None, "add"): [alpha, beta],
        ("add", False, "add", "add"): [alpha, beta, gamma],
        ("add", False, "mul", "add"): [alpha, beta, gamma],
        ("add", True, None, "add"): [alpha, beta, phi],
        ("add", True, "add", "add"): [alpha, beta, gamma, phi],
        ("add", True, "mul", "add"): [alpha, beta, gamma, phi],
        (None, False, None, "mul"): [alpha],
        (None, False, "add", "mul"): [alpha, gamma],
        (None, False, "mul", "mul"):  [alpha, gamma],
        ("add", False, None, "mul"): [alpha, beta],
        ("add", False, "add", "mul"): [alpha, beta, gamma],
        ("add", False, "mul", "mul"): [alpha, beta, gamma],
        ("add", True, None, "mul"): [alpha, beta, phi],
        ("add", True, "add", "mul"): [alpha, beta, gamma, phi],
        ("add", True, "mul", "mul"): [alpha, beta, gamma, phi]
    }
    model_params = params[(trend, damped, seasonal, error_type)]
    seasonals = np.zeros(seasonal_periods+1).tolist() # Seasonal components are initialized as zero.
    
    # Flattening the components and parameters to use in optimization function.
    init_components = [data.float().mean()] + seasonals + [seasonal_periods] 
    init_values = init_components + model_params

    def log_likelihood(init_values,data,seasonal_periods,model):
        # Split the initial values to components, seasonal components and parameters.
        components = init_values[:2].tolist()
        seasonal = init_values[2:2+seasonal_periods].tolist()
        params = init_values[3+seasonal_periods:]
        components = components + [seasonal] + [seasonal_periods]

        # Calculate the log likelihood of the model.
        errors = model(data,init_components=components,params=params)
        log_likelihood = np.sum(norm.logpdf(errors, loc=0, scale=1))
        return -log_likelihood

    # Use the optimization function to find the maximum likelihood estimates with random initial values
    result_params = opt.minimize(log_likelihood,init_values, args=(data,seasonal_periods,selected_model,), method='L-BFGS-B')

    def parse_results(result_params):
        # Unflatten the components and parameters.
        init_components = result_params.x[:2].tolist()
        seasonal = result_params.x[2:2+seasonal_periods].tolist()
        params = result_params.x[3+seasonal_periods:]
        init_components = init_components + [seasonal] + [seasonal_periods]
        return init_components, params
    
    result_components, result_params = parse_results(result_params)

    return result_components, result_params