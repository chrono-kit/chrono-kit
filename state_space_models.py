import numpy as np
import pandas as pd
import torch
from model import Model
from dataloader import DataLoader
import scipy.stats as stats

class ETS_ANN(Model):

    def __init__(self, dep_var, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        super().set_allowed_kwargs(["alpha", "conf"])
        super().__init__(dep_var, indep_var, **kwargs)

        if "alpha" not in kwargs:
            self.alpha = self.__estimate_alpha()
        
        else:

            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])

    def __calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""

        #The implementation of stats.norm.ppf function without using scipy
        #can be left for later

        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __estimate_alpha(self):
        """Estimate alpha to use in equations if not provided by the user"""
        #LEFT EMPTY FOR LATER IMPLEMENTATION
        return torch.tensor(0.5, dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        


    def __get_confidence_interval(self, h):

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self, lprev):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.alpha, self.error))
    
    def __smooth_error(self, y, lprev):

        """Calculate error"""

        self.error = torch.sub(y, lprev)

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} + e_t
        l_t = l_{t-1} + alpha*e_t
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = row[0]
                self.error = torch.tensor(0, dtype=torch.float32)
                self.fitted[0] = row
            
            else:

                self.__smooth_error(row, self.level)
                self.__smooth_level(self.level)
                self.fitted[index] = self.level
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        if return_confidence:

            self.forecast = self.level
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):

                step_forecast = self.level

                self.forecast = torch.cat((self.forecast, step_forecast))

                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

            return self.forecast, (upper_bounds, lower_bounds)
        

        else:

            self.forecast = self.level
            for i in range(1,h):
                self.forecast = torch.cat((self.forecast, self.level))

            return self.forecast


class ETS_MNN(Model):

    def __init__(self, dep_var, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        super().set_allowed_kwargs(["alpha", "conf"])
        super().__init__(dep_var, indep_var, **kwargs)

        if "alpha" not in kwargs:
            self.alpha = self.__estimate_alpha()
        
        else:

            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])

    def __calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""

        #The implementation of stats.norm.ppf function without using scipy
        #can be left for later

        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __estimate_alpha(self):
        """Estimate alpha to use in equations if not provided by the user"""
        #LEFT EMPTY FOR LATER IMPLEMENTATION
        return torch.tensor(0.5, dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        

    def __get_confidence_interval(self, h):

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self, lprev):
        """Calculate the level"""

        self.level = torch.multiply(lprev, torch.add(1, torch.multiply(self.alpha, self.error)))
    
    def __smooth_error(self, y, lprev):

        """Calculate error"""

        self.error = torch.divide(torch.sub(y, lprev), lprev)

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1}*(1 + e_t)
        l_t = l_{t-1}*(1 + alpha*e_t)
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = row[0]
                self.error = torch.tensor(0, dtype=torch.float32)
                self.fitted[0] = row
            
            else:

                self.__smooth_error(row, self.level)
                self.__smooth_level(self.level)
                self.fitted[index] = self.level
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
           This is again the same as the simple exponential smoothing forecast"""

        if return_confidence:

            self.forecast = self.level
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):
      
                step_forecast = self.level

                self.forecast = torch.cat((self.forecast, step_forecast))

                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

            return self.forecast, (upper_bounds, lower_bounds)
        

        else:

            self.forecast = self.level
            for i in range(1,h):
                self.forecast = torch.cat((self.forecast, self.level))

            return self.forecast
