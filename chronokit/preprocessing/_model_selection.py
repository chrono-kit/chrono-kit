import numpy as np
import torch
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.exponential_smoothing._ets import ETS 
from chronokit.exponential_smoothing._smoothing import ExponentialSmoothing
from chronokit.arima._arima import ARIMA

def AIC(log_likelihood, num_parameters):

    return -2*np.log(log_likelihood) + 2*num_parameters

def AIC_corrected(log_likelihood, num_parameters, num_observations):

    factor = 2*num_parameters*(num_parameters+1)/(num_observations-num_parameters-1)

    return AIC(log_likelihood, num_parameters) + factor

def BIC(log_likelihood, num_parameters, num_observations):

    factor = num_parameters*(np.log(num_observations) - 2)

    return AIC(log_likelihood, num_parameters) + factor


def model_selection(data, model, criterion="aic", seasonal_periods=None, max_order=3):

    """
    Function for model selection by using information criterions

    Arguments:

    *data (array_like): Time series data to use for model selection
    *model (str): Type of model to run selection for. "smoothing" or "arima".
    *criterion Optional[str]: Information criterion to use for selection. "aic", "bic" or "aic_c"
    *seasonal_periods Optional[int]: Seasonality period to use for models. If model="smoothing", it is required. Ignored otherwise.
    *max_order Optional[int]: Maximum order of p and q to use for estimating ARIMA order. Default = 3.

    Returns:

    *results (dict): Criterion results of the selection.
    *params (dict): Recommended model parameters.
    """

    assert(model in ["smoothing", "arima"]), "Model not accepted"
    assert(criterion in ["aic", "bic", "aic_c"]), "Criterion not accepted"

    def log_likelihood(errors):
        err = DataLoader(errors).to_numpy()
        err = err[~np.isnan(err)]
        loc = np.nanmean(err)
        scale = np.nanstd(err)**2
        log_likelihood = np.sum(norm.logpdf(err, loc=loc, scale=scale))
        return -log_likelihood

    if model == "smoothing":
        assert(type(seasonal_periods)==int and seasonal_periods >= 2), "Wrong seasonality period"

        trend = ["add", None]
        seasonal = ["add", "mul", None]
        damped = [True, False]
        error_type = ["add", "mul", None]
        
        #### TO DO ####
        #Write a better algorithm instead of nested for loops

        results = {}
        params = {"trend": None, "seasonal": None, "damped": None, "error": None}
        min_val = 1e6
        for e in error_type:    
            if e is None:
                trend_seasonal = [(None, None), ("add", None),
                          ("add", "add"), ("add", "mul")]
                for t,s  in trend_seasonal:
                    if s == "mul":
                        if DataLoader(data).to_numpy().min() > 0:
                            continue
                    else:
                        for d in damped:
                            model = ExponentialSmoothing(data, trend=t, seasonal=s, seasonal_periods=seasonal_periods, damped=d)
                            model.fit()
                            errors = DataLoader(data).to_tensor() - model.fitted
                            ll = log_likelihood(errors)

                            k = sum([bool(t), bool(s), d]) + 1 #plus 1 for the level parameter

                            if criterion == "aic":
                                res = AIC(ll, k)
                            elif criterion == "aic_c":
                                res = AIC_corrected(ll, k, len(model.data))
                            elif criterion == "bic":
                                res = BIC(ll, k, len(model.data))
                                
                            name = str(model.__class__).split(".")[-1].replace("'>", "")
                            results[f"{name}, damped={d}"] = res

                            if res < min_val:
                                params["trend"] = t
                                params["seasonal"] = s
                                params["damped"] = d
                                params["error"] = e
                                min_val = res
            else:
                trend_seasonal = [(None, None), 
                                  (None, "add"),
                                  (None, "mul"),
                                  ("add", None),
                                  ("add", "add"),
                                  ("add", "mul")]
                for t,s  in trend_seasonal:
                    if s == "mul":
                        if DataLoader(data).to_numpy().min() > 0:
                            continue
                    else:
                        for d in damped:
                            try:
                                model = ETS(data,error_type=e, trend=t, seasonal=s, seasonal_periods=seasonal_periods, damped=d)
                                model.fit()
                                errors = DataLoader(data).to_tensor() - model.fitted
                                ll = log_likelihood(errors)

                                k = sum([bool(t), bool(s), d]) + 1 #plus 1 for the level parameter

                                if criterion == "aic":
                                    res = AIC(ll, k)
                                elif criterion == "aic_c":
                                    res = AIC_corrected(ll, k, len(model.data))
                                elif criterion == "bic":
                                    res = BIC(ll, k, len(model.data))
                                    
                                name = str(model.__class__).split(".")[-1].replace("'>", "")
                                results[f"{name}, damped={d}"] = res

                                if res < min_val:
                                    params["trend"] = t
                                    params["seasonal"] = s
                                    params["damped"] = d
                                    params["error"] = e
                                    min_val = res
                            except:
                                pass
        
    elif model == "arima":
        assert(type(max_order) == int and max_order > 0), "Argument max_order must be an integer gretar than 0"
        orders = [(p,d,q) for p in range(max_order+1) for d in range(2) for q in range(max_order+1) if (p,d,q) not in [(0,0,0), (0,1,0)]]
        
        #### TO DO ####
        #Write a better algorithm instead of nested for loops

        results = {}
        params = {"p": None, "d": None, "q": None}
        min_val = 1e6
        for order in orders:    
            
            try:
                model = ARIMA(data, p=order[0], d=order[1], q=order[2])
                model.fit()
                errors = torch.tensor(model.errors).detach().clone()
                ll = log_likelihood(errors)

                k = sum(order)

                if criterion == "aic":
                    res = AIC(ll, k)
                elif criterion == "aic_c":
                    res = AIC_corrected(ll, k, len(model.data))
                elif criterion == "bic":
                    res = BIC(ll, k, len(model.data))
                                    
                results[f"Order: {order}"] = res

                if res < min_val:
                    params["p"] = order[0]
                    params["d"] = order[1]
                    params["q"] = order[2]
                    min_val = res
            except:
                pass

    return results, params





