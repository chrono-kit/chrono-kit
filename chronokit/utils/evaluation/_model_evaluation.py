import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader

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

class ModelEvaluator:

    def __init__(
                self, 
                eval_metrics = ["mae"],
                criterions=["aic"],
):
        """
        Class for evaluation time series model performance.

        The functionalities inside this class were written
        as Hyndman, R. et al.[1] Chapter 7 as reference.

        Arguments:

        *eval_metrics Optional([list]): List of evaluation metrics to use.
            One or more of; 
            ["mae", "mse", "rmse", "mase", "mape", "s_mape"]
        
        *criterions Optional([list]): List of information criterions to use.
            One or more of; 
            ["aic", "aic_c", "bic", "hqic"]

        [1]'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        """
        valid_ic = {
                    "aic": AIC,
                    "aic_c": AIC_corrected,
                    "bic": BIC,
                    "hqic": HQIC
        }
        
        valid_metrics = {
                        "mae": mae,
                        "mse": mse,
                        "rmse": rmse,
                        "mape": mape,
                        "s_mape": symmetric_mape,
                        "mase": mase,
        }


        if not isinstance(criterions, list):
            if isinstance(criterions, str) and criterions in valid_ic:
                criterions = [criterions]
            else:
                criterions = ["aic"]
        
        if not isinstance(eval_metrics, list):
            if isinstance(eval_metrics, str) and eval_metrics in valid_metrics:
                eval_metrics = [eval_metrics]
            else:
                eval_metrics = ["mae"]

        self.ic = {
                    k.upper() if k != "aic_c" else "AIC_c" : valid_ic[k] 
                    for k in criterions
        }

        self.metrics = {
                    k.upper() if k != "s_mape" else "sMAPE" : valid_metrics[k] 
                    for k in eval_metrics
        }

    def evaluate_model(self, model, val_data):
        """
        Evaluate the given model's performance on given data
        
        2 evaluation steps are performed;

        1) Evaluate model fit by selected information criterions
        2) Evaluate model predictions by selected metrics

        Arguments:

        *model (chronokit.TraditionalTimeSeriesModel): Model
            to perform evaluation for. Must be fitted before
            evaluation
        *val_data (array_like): Data to perform evaluation
            on

        Returns:

        *model_results (dict): Dictionary containing evaluation
            results for the model
        """

        if torch.nansum(abs(model.fitted)) == 0:
            raise Warning(
                "Please call .fit() before evaluation"
            )

        def log_likelihood(errors):
            err = DataLoader(errors).match_dims(1, return_type="numpy")
            err = err[~np.isnan(err)]
            loc = np.nanmean(err)
            scale = np.nanstd(err)
            log_likelihood = np.sum(norm.logpdf(err, loc=loc, scale=scale))
            return log_likelihood

        errors = model.data - model.fitted
        
        ll = log_likelihood(errors)
        num_observations = len(model.fitted)
        num_parameters = model.info["num_params"]
        
        forecasts = model.predict(h=len(val_data), confidence=None)

        model_results = {}

        val_data = DataLoader(val_data).match_dims(1, return_type="tensor")
        forecasts = DataLoader(forecasts).match_dims(1, return_type="tensor")

        for criterion, func in self.ic.items():
            #AIC function does not take num_observations argument
            if criterion == "AIC":
                ic = func(ll, num_parameters)
            else:
                ic = func(ll, num_parameters, num_observations)

            model_results[criterion] = ic.item()

        for metric, func in self.metrics.items():
            res = func(y_pred=forecasts, y_true=val_data)

            model_results[f"Val_{metric}"] = res.item()

        return model_results
    
    def run_model_selection(
                            self,
                            data,
                            model_type,
                            val_data = None, 
                            seasonal_periods=None, 
                            smoothing_args = {
                                            "error": [None, "mul", "add"],
                                            "trend": [None, "mul", "add"],
                                            "seasonal": [None, "mul", "add"],
                                            "damped": [True, False]
                            },
                            sarima_args = {
                                        "max_p": 2, "min_p": 1, 
                                        "max_d": 1, "min_d": 0,
                                        "max_q": 2, "min_q": 0, 
                                        "max_P": 1, "min_P": 1,
                                        "max_D": 1, "min_D": 0, 
                                        "max_Q": 1, "min_Q": 0
                            }
):

        """
        Run a model selection procedure with specified arguments for help
        select best model for the given data.

        The procedure is as follows;

        1) Split data into training and validation
            * If val_data is not None;
                training = data
                validation = val_data
            * Else:
                * If seasonal_periods is Not None;
                    training = data[:-seasonal_periods]
                    validation = data[-seasonal_periods:]

                * Else:
                    training = data[:-3]
                    validation = data[-3:]

        2) Generate models specified by the arguments and estimate parameters
        using the training data

        3) Calculate Information Criterion values on model fits. Then make predictions
        and calculate regression metric values on validation data.

        4) Rank the models by the summing the calcualted criterion and metric values.

        5) The best model is the one that achieved the minimum value.

        For more information; refer to Hyndman, R. et al. Chapter 7

        Arguments:

        *data (array_like): Univariate time series data to use for 
            parameter estimation for the generated models
        *model_type (str): Type of models to run selection for.
            One of; ["smoothing", "arima", "all"]. If "all",
            both smoothing and arima models are compared
        *val_data (Optional[array_like]): Validation data to use
            for evaluating model predictions. data argument will
            be split into two if val_data = None
        *seasonal_periods (Optional[int]): Seasonality period
            if seasonal models should be incorporated to selection
        *smoothing_args (Optional[dict]): Model arguments for smoothing
            models. Will generate smoothing models with arguments that
            are present in smoothing_args dict only. Default;
                {
                "error": [None, "mul", "add"],
                "trend": [None, "mul", "add"],
                "seasonal": [None, "mul", "add"],
                "damped": [True, False]
                }
        *sarima_args (Optional[dict]): Model arguments for SARIMA
            models. Will generate SARIMA models with arguments that
            are present in smoothing_args dict only. Default;
                {
                "max_p": 2, "min_p": 1, 
                "max_d": 1, "min_d": 0,
                "max_q": 2, "min_q": 0, 
                "max_P": 1, "min_P": 1,
                "max_D": 1, "min_D": 0, 
                "max_Q": 1, "min_Q": 0
                }

        Returns:

        *results (pandas.DataFrame): Numerical results for evaluated models
            for each information criterion and evaluation metric
        *rankings (list): Ordered rankings of the models with the suggested
            selection (sum of all ic and metric values). First index is the best
            model and the last index is the worst
        """

        assert model_type in ["smoothing", "arima", "all"], "Model_type not accepted"

        data = DataLoader(data).match_dims(dims=1, return_type="tensor")

        assert (len(data) > 10), "Number of observations must be > 10 \
            for a meaningful model selection procedure"

        if val_data is None:

            assert (len(data) > 13), "Number of observations must be > 13 \
            for a meaningful model selection procedure if val_data = None"

            if seasonal_periods is not None:
                if len(data) > seasonal_periods*3:
                    tr_data = data[:-seasonal_periods].clone()
                    val_data = data[-seasonal_periods:].clone()
                elif len(data) > 2*seasonal_periods + 3:
                    tr_data = data[:-3].clone()
                    val_data = data[-3:].clone()
                else:
                    raise ValueError(
                        f"Number of observations must be > 2*seasonal_periods + 3 \
                        if val_data = None. Got nobs={len(data)} <= {2*seasonal_periods + 3}"
                    )
            else:
                tr_data = data[:-3].clone()
                val_data = data[-3:].clone()
        else:
            tr_data = data.clone()
            val_data = DataLoader(val_data).match_dims(dims=1, return_type="tensor")
                    
        
        if model_type == "smoothing":
            print("Evaluating parameters for smoothing models...")
            models = self.__get_smoothing_models(tr_data, seasonal_periods, args=smoothing_args)
            
        elif model_type == "arima":
            print("Evaluating parameters for arima models...")
            models = self.__get_sarima_models(tr_data, seasonal_periods, args=sarima_args)

        else:
            print("Evaluating parameters for smoothing models...")
            smoothing_models = self.__get_smoothing_models(tr_data, seasonal_periods, args=smoothing_args)

            print("Evaluating parameters for arima models...")
            arima_models = self.__get_sarima_models(tr_data, seasonal_periods, args=sarima_args)

            models = smoothing_models + arima_models

        results = {}

        for model in models:

            model.fit()
            eval_results = self.evaluate_model(model, val_data)

            model_class = model.info["model"]

            results[model_class] = eval_results
        
        results = pd.DataFrame(results).transpose()
        rankings = results.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
        rankings = results.sum(axis=1).sort_values().index.to_list()
        
        print(f"Best Model: {rankings[0]}")

        return results, rankings
                
    def __get_smoothing_models(self, data, seasonal_periods, args):
        """Generate all smoothing models with specified args"""
        from chronokit.exponential_smoothing._ets import ETS
        from chronokit.exponential_smoothing._smoothing import ExponentialSmoothing

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
        """Generate all sarima models with specified args"""
        from chronokit.arima._sarima import SARIMA
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
            start = args[f"min_{component}"] if f"min_{component}" in args else 0
            if f"max_{component}" in args:
                stop = args[f"max_{component}"] + 1
            else:
                if component in ["d", "D"]:
                    stop=2
                else:
                    stop = 4
            
            #Check valid start,stop
            if stop < 0:
                if component in ["d", "D"]:
                    stop=2
                else:
                    stop = 4

            if start > stop:
                start = stop

            provided_args = list(range(start, stop))
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
