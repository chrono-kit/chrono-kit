import numpy as np
import torch
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.exponential_smoothing._ets import ETS
from chronokit.exponential_smoothing._smoothing import ExponentialSmoothing
from chronokit.arima._sarima import SARIMA
from chronokit.utils.evaluation.metrics import (rmse, 
                                                mse, 
                                                mae,
                                                mase,
                                                mape,
                                                symmetric_mape, 
                                                AIC, 
                                                AIC_corrected, 
                                                BIC,
                                                HQIC) 

class ModelSelection:

    def __init__(self, data, split_length, validation_data=None, criterion="aic", **kwargs):

        self.split_length = split_length
        self.data = DataLoader(data).to_tensor()
        self.criterion = {"aic": AIC,
                          "aic_c": AIC_corrected,
                          "bic": BIC,
                          "hqic": HQIC}[criterion]

    def __one_model_procedure(self, model):

        def log_likelihood(errors):
            err = DataLoader(errors).to_numpy()
            err = err[~np.isnan(err)]
            loc = np.nanmean(err)
            scale = np.nanstd(err)
            log_likelihood = np.sum(norm.logpdf(err, loc=loc, scale=scale))
            return -log_likelihood

        model.fit()
        errors = model.data - model.fitted
        
        ll = log_likelihood(errors)
        num_observations = len(model.fitted)
        num_parameters = model.info["num_params"]
        
        ic = self.criterion(ll, num_parameters, num_observations)

        return ic
    
    def run_selection(self, 
                      model_type, 
                      seasonal_periods=None, 
                      smoothing_args = {"error": [None, "mul", "add"],
                                        "trend": [None, "mul", "add"],
                                        "seasonal": [None, "mul", "add"],
                                        "damped": [True, False]},
                      sarima_args = {"max_p": 3, "max_d": 1, "max_q": 3,
                                     "max_P": 3, "max_D": 1, "max_Q": 3}
):

        assert model_type in ["smoothing", "arima", "all"], "Model_type not accepted"

        tr_data = self.data[:-self.split_length]

        if model_type == "smoothing":
            
            models = self.__get_smoothing_models(tr_data, seasonal_periods, args=smoothing_args)
            
        elif model_type == "arima":
            models = self.__get_sarima_models(tr_data, seasonal_periods, args=sarima_args)

        else:
            smoothing_models = self.__get_smoothing_models(tr_data, seasonal_periods, args=smoothing_args)
            arima_models = self.__get_sarima_models(tr_data, seasonal_periods, args=sarima_args)

            models = smoothing_models + arima_models

        results = {}

        for model in models:
            
            ic = self.__one_model_procedure(model)

            model_class = model.info["model"]

            results[model_class] = ic

        return results
                
    def __get_smoothing_models(self, data, seasonal_periods, args):

        if seasonal_periods is not None:
            assert (
                isinstance(seasonal_periods, int) and seasonal_periods >= 2
            ), "Provide seasonal_periods as an integer >= 2"

        assert(isinstance(args, dict)), "Provide smoothing_args as a dictionary"

        models = []

        has_non_positive = bool(torch.min(data).item() < 1e-6)

        model_args = {"error": [], "trend": [], "seasonal": [], "damped":[]}

        for component in model_args.keys():
            if component in args:
                provided_args = args[component]
            else:
                if component != "damped":
                    provided_args = [None, "mul", "add"]
                else:
                    provided_args = [True, False]
            
            if has_non_positive:
               provided_args = [x for x in provided_args if x != "mul"]

            model_args[component] = provided_args     

        trend_args = [
                    (t, d)
                    for t in model_args["trend"]
                    for d in model_args["damped"]]
        trend_args = [x for x in trend_args if x != (None, True)]

        error_args = model_args["error"]
        seasonal_args = model_args["seasonal"] if seasonal_periods is not None else [None]

        selection_args = [
                [e, t[0], s, t[1], seasonal_periods] 
                for e in error_args 
                for t in trend_args 
                for s in seasonal_args
        ]

        for arg in selection_args:
            e,t,s,d,s_per = arg

            if e is None:
                models.append(ExponentialSmoothing(data=data, trend=t, seasonal=s, seasonal_periods=s_per, damped=d,
                                                       initialization_method="heuristic"))
            else:
                models.append(ETS(data=data, error_type=e, trend=t, seasonal=s, seasonal_periods=s_per, damped=d,
                                    initialization_method="heuristic"))
        
        return models
    
    def __get_sarima_models(self, data, seasonal_periods, args):

        if seasonal_periods is not None:
            assert (
                isinstance(seasonal_periods, int) and seasonal_periods >= 2
            ), "Provide seasonal_periods as an integer >= 2"
        
        assert(isinstance(args, dict)), "Provide smoothing_args as a dictionary"

        for k,v in args.items():
            assert (
                isinstance(v, int) and v >= 0
            ), f"Argument {k} must be an integer >= 0"

        models = []

        model_args = {"p": [], "d": [], "q": [], 
                      "P":[], "D": [], "Q": []}

        for component in model_args.keys():
            if f"max_{component}" in args:
                provided_args = list(range(args[f"max_{component}"]+1))
            else:
                if component in ["d", "D"]:
                    provided_args = list(range(2))
                else:
                    provided_args = list(range(4))

            model_args[component] = provided_args 

        orders = [
            (p, d, q)
            for p in model_args["p"]
            for d in model_args["d"]
            for q in model_args["q"]
        ]
            
        if seasonal_periods is None:
            seasonal_orders = [(0,0,0)]
        else:
            seasonal_orders = [
                            (p, d, q)
                            for p in model_args["P"]
                            for d in model_args["D"]
                            for q in model_args["Q"]
            ]

        selection_args = [
            (pdq, PDQ) 
            for pdq in orders
            for PDQ in seasonal_orders
        ]

        selection_args = [
            ord for ord in selection_args 
            if ord not in [
                ((0,d1,0),(0,d2,0))
                for d1 in range(max(model_args["d"]) + 1)
                for d2 in range(max(model_args["D"]) + 1)
            ]
        ]                
            
        for orders in selection_args:
            order, seasonal_order = orders

            models.append(SARIMA(data=data, order=order, seasonal_order=seasonal_order,
                                     seasonal_periods=seasonal_periods))
            
        return models
