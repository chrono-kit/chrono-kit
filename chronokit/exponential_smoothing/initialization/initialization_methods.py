import numpy as np
import pandas as pd
import torch
import scipy.optimize as opt
from scipy.stats import norm
from scipy.stats import linregress
from chronokit.exponential_smoothing.initialization import ets_methods, smoothing_methods
from chronokit.preprocessing.dataloader import DataLoader

def get_init_method(method):

    init_method = {"heuristic": heuristic_initialization,
                    "mle": mle_initialization}[method]
    
    return init_method

def get_smooth_method(error, trend, seasonal, damped):

    if error:
        smooth_method = { 
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
                        }[trend, damped, seasonal, error]
    
    else:
        smooth_method = { 
                        (None, False, None): smoothing_methods.simple_exp,
                        ("add", False, None): smoothing_methods.holt_trend,
                        ("add", True, None): smoothing_methods.holt_damped_trend,
                        ("add", False, "add"): smoothing_methods.hw_add,
                        ("add", False, "mul"): smoothing_methods.hw_mul,
                        ("add", True, "add"): smoothing_methods.hw_damped_add,
                        ("add", True, "mul"): smoothing_methods.hw_damped_mul
                        }[trend, damped, seasonal]
        
    return smooth_method

def heuristic_initialization(data, trend=False, seasonal=None,
                              seasonal_periods=None):
    """
    Heuristic initialization method for initial components
    See: Hyndman et al. section 2.6.1
    """

    data = DataLoader(data).to_numpy().copy()

    if data.ndim >= 1:

        data = np.squeeze(data)

    initial_level = None
    initial_trend = None
    initial_seasonal = None

    assert(len(data) >= 10), "Length of data must be >= for heuristic initialization"

    if seasonal:
        
        #Data must have at least 2 full seasonal cycles
        assert (len(data) > 2*seasonal_periods), "Length of data must be > 2*seasonal_periods"

        #Max number of seasonal cycles to be used is 5
        seasonal_cycles = min(5, len(data)//seasonal_periods)
        series = pd.Series(data[:seasonal_periods*seasonal_cycles])
        moving_avg = series.rolling(seasonal_periods, center=True).mean()

        if seasonal_periods % 2 == 0:
            moving_avg = moving_avg.shift(-1).rolling(2).mean()

        if seasonal == "add":
            detrend = series - moving_avg
        elif seasonal == "mul":
            detrend = series/moving_avg
     
        initial_seasonal = np.zeros(shape=(seasonal_periods,seasonal_cycles))*np.nan
        for i in range(0, seasonal_cycles):
            initial_seasonal[:, i] = detrend[i*seasonal_periods:(i+1)*seasonal_periods]
    
        initial_seasonal = np.nanmean(initial_seasonal, axis=1)

        if seasonal == "add":
            #normalize so that the sum is equal to 1
            initial_seasonal /= np.sum(initial_seasonal)
        elif seasonal == "mul":
            #normalize so that the sum is equal to m=seasonal_periods
            initial_seasonal *= seasonal_periods/np.sum(initial_seasonal)

        adjusted_data = moving_avg.dropna().values
    
    else:
        adjusted_data = data.copy()

    
    result = linregress(x=np.arange(10), y=adjusted_data[:10])
    initial_level = result[1]

    if trend:
        initial_trend = result[0]

    return initial_level, initial_trend, initial_seasonal


def mle_initialization(data, trend=None,damped=False, seasonal=None, error_type=None,seasonal_periods=12,alpha=0.1,beta=0.01,gamma=0.01,phi=0.99):
    """Maximum Likelihood Estimatin for initialization of initial components"""
 
    data = DataLoader(data).to_numpy().copy()
    selected_model = get_smooth_method(error=error_type, trend=trend, seasonal=seasonal, damped=damped)

    # Find the parameters of selected model to maximize the likelihood function.
    params = {
        (None, False, None): [alpha],
        (None, False, "add"): [alpha, gamma],
        (None, False, "mul"): [alpha, gamma],
        ("add", False, None): [alpha, beta],
        ("add", False, "add"): [alpha, beta, gamma],
        ("add", False, "mul"): [alpha, beta, gamma],
        ("add", True, None): [alpha, beta, phi],
        ("add", True, "add"): [alpha, beta, gamma, phi],
        ("add", True, "mul"): [alpha, beta, gamma, phi],
    }
    model_params = params[(trend, damped, seasonal)]
    seasonals = np.zeros(seasonal_periods+1).tolist() # Seasonal components are initialized as zero.
    
    # Flattening the components and parameters to use in optimization function.
    init_components = [data.mean(), 0] #level and trend
    if seasonal:
        for i in range(seasonal_periods):
            init_components.append(1) #initialize seasonals as 1
        
    init_components.append(seasonal_periods)
    init_values = init_components + model_params

    def log_likelihood(init_values,data,seasonal_periods,model):
        # Split the initial values to components, seasonal components and parameters.
        components = init_values[:2].tolist()
        seasonal_components = init_values[2:-len(model_params)].tolist()
        params = init_values[-len(model_params):]
        components = components + [seasonal_components] + [seasonal_periods]

        dep_var = DataLoader(data).to_tensor().detach().clone()
        if dep_var.ndim == 1:
            dep_var = torch.unsqueeze(dep_var, axis=1)
        # Calculate the log likelihood of the model.
        errors = model(dep_var,init_components=components,params=params)
        loc = np.array(errors).mean()
        scale = np.array(errors).std()**2
        log_likelihood = np.sum(norm.logpdf(errors, loc=loc, scale=scale))
        return -log_likelihood

    # Use the optimization function to find the maximum likelihood estimates with random initial values
    result_params = opt.minimize(log_likelihood,init_values, args=(data,seasonal_periods,selected_model,))

    def parse_results(result_params):
        # Unflatten the components and parameters.
        results = list(result_params.x)
        initial_level, initial_trend = results[:2]
        initial_seasonal = np.array(results[2:-len(model_params)-1])
        if not trend:
            initial_trend = None
        if not seasonal:
            initial_seasonal = None
        init_components = (initial_level, initial_trend, initial_seasonal)
        return init_components
    
    result_components = parse_results(result_params)

    return result_components