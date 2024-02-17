import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.style as style
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.utils.evaluation._model_evaluation import ModelEvaluator
from chronokit.utils.vis_utils.model_plots import (plot_predictions,
                                                   plot_model_fit
)

class TraditionalTimeSeriesModel:

    def __init__(self, data, **kwargs):
        """
        Base model class for all model classes to inherit from.
        Child classes are expected to implement their own .fit()
        and .predict() methods

        As of v1.1.x, inherits itself to two model classes;

        * ARIMAProcess : Which is written with the book [1] as reference
        * InnovationsStateSpace : Which is written with the book [2] as reference

        [1]: 'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
            Time series analysis: Forecasting and control (5th ed).'

        [2]: 'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        """

        self.data_loader = DataLoader(data)

        #Assert Valid Data
        try:
            self.data = self.data_loader.match_dims(1, return_type="torch")
        except:  # noqa: E722
            raise NotImplementedError(
                "Multivariate models are not implemented as of v1.1.0,\
                please make sure data.ndim <= 1 or squeezable"
            )

        self.info = {}

    def evaluate(
            self,
            val_data,
            metrics=["mae", "rmse", "mase"],
            criterions=["aic", "bic", "hqic"]
):
        """
        Evaluate the performance of the fitted model
        with the given validation data

        2 evaluation steps are performed;

        1) Evaluate model fit by selected information criterions
        2) Evaluate model predictions by selected metrics

        Plots model fit/predictions and returns metrics

        Arguments:

        *val_data (array_like): Data to perform evaluation on
        *metrics Optional(list): Regression metrics for evaluation
        *criterions Optional(list): Information criterions to
            evaluate model fit

        Returns:

        *res (pandas.DataFrame): Numerical results of the evaluation
        """

        if torch.nansum(abs(self.fitted)) == 0:
            raise Warning(
                "Please call .fit() before evaluation"
            )

        evaluator = ModelEvaluator(eval_metrics=metrics, criterions=criterions)

        val_data = DataLoader(val_data).match_dims(dims=1, return_type="tensor")

        res = evaluator.evaluate_model(self, val_data)
        res = pd.DataFrame(data=res.values(), index=res.keys(), columns=["Value"])

        forecasts, bounds = self.predict(len(val_data), confidence=0.95)

        if len(self.data) > 2*len(val_data):
            pre_vals = self.data[-2*len(val_data):].clone()
        else:
            pre_vals = self.data.clone()

        style.use("ggplot")
        fig, axes = plt.subplots(2, 1, figsize=(16,8))

        for ind, ax in enumerate(axes):
            if ind == 0:
                plot_model_fit(self, title="True Data vs. Model Fit", ax=ax)
                               
            else:
                plot_predictions(y_true=val_data, y_pred=forecasts, bounds=bounds, 
                         pre_vals=pre_vals, title="Prediction vs Val Data(%95 Confidence)", ax=ax)

        plt.show()

        return res

    def _confidence_interval_simulation_method(self, h, confidence):
        """Refer to Hyndman et al. Chapter 6.1"""
        M = 5000 #AS recommended by Hyndman et. al
        q1 = (1-confidence)/2
        q2 = 1 - q1

        fit_errors = self.data - self.fitted

        #torch.nanstd() has not been implemented yet
        #we calculate in numpy then convert to torch tensors
        loc = torch.tensor(np.nanmean(fit_errors.clone().numpy()), dtype=torch.float32)
        scale = torch.tensor(np.nanstd(fit_errors.clone().numpy()), dtype=torch.float32)

        sample_errors = torch.normal(loc, scale, size=(M,h))

        q1_sample = torch.quantile(sample_errors, q1, dim=0, interpolation="nearest")
        q2_sample = torch.quantile(sample_errors, q2, dim=0, interpolation="nearest")

        bounds = torch.abs(torch.sub(q1_sample, q2_sample))

        return bounds
    
    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        valid_kwargs = self.check_kwargs(kwargs)
        for k, v in valid_kwargs.items():
            self.__setattr__(k, v)
    
    def check_kwargs(self, kwargs: dict):
        return kwargs
    
    def fit(self):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, h: int, confidence: float = None):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

class NeuralTimeSeriesModel:
    '''
    Torch implementation of the NARX model.
    '''
    def __init__(
                self,
                data,
                exogenous_data=None,
                **kwargs,
):
        #super(NeuralTimeSeriesModel, self).__init__()

        self.data_loader = DataLoader(data)

        try:
            self.data = self.data_loader.match_dims(dims=1, return_type="tensor")

        except: #noqa: E722
            raise NotImplementedError(
                "Multivariate models are not implemented as of v1.1.0,\
                please make sure data.ndim <= 1 or squeezable"
            )

        if exogenous_data is not None:
            try:
                self.exogenous_data = DataLoader(exogenous_data).match_dims(dims=1, return_type="tensor")
            except: #noqa: E722
                raise NotImplementedError(
                    "Multivariate models are not implemented as of v1.1.0,\
                    please make sure exogenous data.ndim <= 1 or squeezable"
                )

            if len(self.exogenous_data) != len(self.data):
                raise NotImplementedError(
                    f"Models with exogenous data of different length than \
                    endogenous data have not been implemented as of v1.1.x.\n\
                    Got {len(self.exogenous_data)} != {len(self.data)}"
                )
            
        else:
            self.exogenous_data = None
        
        self.info = {}

    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        valid_kwargs = self.check_kwargs(kwargs)
        for k, v in valid_kwargs.items():
            self.__setattr__(k, v)
    
    def check_kwargs(self, kwargs: dict):
        return kwargs