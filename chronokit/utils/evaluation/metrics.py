import torch
from chronokit.preprocessing._dataloader import DataLoader

"""
Performance evaluation metrics for model predictions
And Information Criterion for assessing model performance 
during fitting.

All metrics and criterions have been written with 
Hyndman, R. et al. [1] as reference.

[1]'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
    Forecasting with exponential smoothing: The state space approach'
"""


def mae(y_pred, y_true):
    """
    Mean Absolute Error

    Arguments:

    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    err = y_true - y_pred
    err = err[~torch.isnan(err)]

    return torch.mean(torch.abs(err))


def mse(y_pred, y_true):
    """
    Mean Squared Error

    Arguments:

    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    err = y_true - y_pred
    err = err[~torch.isnan(err)]

    return torch.mean((torch.square(err)))


def rmse(y_pred, y_true):
    """
    Root Mean Squared Error

    Arguments:

    *y_pred (array_like): Predicted values
    *y_true (array_like): Ground truth values
    """
    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    err = y_true - y_pred
    err = err[~torch.isnan(err)]

    return torch.sqrt(torch.mean((torch.square(err))))

def mape(y_pred, y_true):
    """
    Mean Absolute Percentage Error
    """

    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    assert (torch.min(abs(y_true)).item() > 1e-6), "MAPE cannot be used for time series data that have\
        values extremely close to 0 (<= 1e-6)"
    err = y_true - y_pred
    err = err[~torch.isnan(err)]
    return torch.mean(abs(100*err/y_true))

def symmetric_mape(y_pred, y_true):
    """
    Symmetric MAPE
    """

    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    assert (torch.min(abs(y_true)).item() > 1e-6), "sMAPE cannot be used for time series data that have\
        values extremely close to 0 (<= 1e-6)"
    
    err = y_true - y_pred
    err = err[~torch.isnan(err)]

    return torch.mean(200*abs(err)/(y_true+y_pred))

def mase(y_true, y_pred):
    """
    Mean Absolute Scaled Error
    """

    y_pred = DataLoader(y_pred).match_dims(dims=1, return_type="tensor")
    y_true = DataLoader(y_true).match_dims(dims=1, return_type="tensor")

    err = y_true - y_pred
    err = err[~torch.isnan(err)]

    scale = torch.mean(abs(y_true[1:]- y_true[:-1]))

    return torch.mean(abs(err/scale))

def IC(log_likelihood, num_parameters, penalty_factor):
    """Default Information Criterion"""
    return -2*log_likelihood + num_parameters*penalty_factor


def AIC(log_likelihood, num_parameters):
    """Akaike Information Criterion"""
    ll = DataLoader(log_likelihood).to_tensor()
    num_parameters = DataLoader(num_parameters).to_tensor()

    return IC(ll, num_parameters, penalty_factor=2)


def AIC_corrected(log_likelihood, num_parameters, num_observations):
    """Corrected Akaike Information Criterion"""
    nobs = DataLoader(num_observations).to_tensor()
    ll = DataLoader(log_likelihood).to_tensor()
    num_parameters = DataLoader(num_parameters).to_tensor()

    penalty_factor = (2*nobs)/(nobs - num_parameters - 1)

    return IC(ll, num_parameters, penalty_factor)


def BIC(log_likelihood, num_parameters, num_observations):
    """Bayesian Information Criterion"""
    nobs = DataLoader(num_observations).to_tensor()
    ll = DataLoader(log_likelihood).to_tensor()
    num_parameters = DataLoader(num_parameters).to_tensor()

    
    penalty_factor = torch.log(nobs)

    return IC(ll, num_parameters, penalty_factor)

def HQIC(log_likelihood, num_parameters, num_observations):
    """Hannan-Quinn Information Criterion"""
    nobs = DataLoader(num_observations).to_tensor()
    ll = DataLoader(log_likelihood).to_tensor()
    num_parameters = DataLoader(num_parameters).to_tensor()
    
    assert (nobs >= 3), "HQIC requires at least 3 observations"
    penalty_factor = 2*torch.log(torch.log(nobs))

    return IC(ll, num_parameters, penalty_factor)