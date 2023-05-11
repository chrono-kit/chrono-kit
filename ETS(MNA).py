import numpy as np
import pandas as pd
import torch
from model import Model
from dataloader import DataLoader
import scipy.stats as stats

class ETS_MNA(Model):

    def __init__(self, dep_var,seasonal_periods=4, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        super().set_allowed_kwargs(["alpha", "gamma","conf"])
        super().__init__(dep_var, indep_var, **kwargs)


        self.damped = True
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        if "alpha" not in kwargs:
            self.alpha = self.__estimate_alpha()
        
        else:

            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        if "gamma" not in kwargs:
            self.gamma = self.__estimate_gamma()
        
        else:

            self.gamma = torch.tensor(self.gamma, dtype=torch.float32)

        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
        
        self.__calculate_conf_level()
        self.seasonal=torch.tensor()
        self.level=torch.tensor() ##?? should it be constant or list?
        self.residuals = torch.tensor([])

    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, seasonal = heuristic_initialization(self.dep_var, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_seasonals = seasonal

        alpha = 0.1
        gamma = 0.01

        
        self.params = self.__estimate_params(init_components = np.array([self.initial_level, self.initial_seasonals, self.seasonal_periods]),
                               params = np.array([alpha, gamma]))
        
        self.alpha,  self.gamma,  = self.params

        
    def __calculate_conf_level(self): ###???????????????

        """Calculate the confidence level to be used for intervals"""

        #The implementation of stats.norm.ppf function without using scipy
        #can be left for later

        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __estimate_alpha(self):
        """Estimate alpha to use in equations if not provided by the user"""
        #LEFT EMPTY FOR LATER IMPLEMENTATION
        return torch.tensor(0.5, dtype=torch.float32)
    
    def __estimate_gamma(self):
        """Estimate gama to use in equations if not provided by the user"""
        #LEFT EMPTY FOR LATER IMPLEMENTATION
        return torch.tensor(0.5, dtype=torch.float32)

    def __update_res_variance(self, error):   ###???????????????

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        
    def __get_confidence_interval(self, h):  ###???????????????

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self,lprev,seasonal,error): ##done
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.alpha, torch.multiply(torch.add(lprev,seasonal),error)))
    
    def __smooth_error(self, y, lprev,seasonal):  # What is the equation??

        """Calculate error"""

        self.error = torch.divide(torch.sub(y,torch.add(seasonal, lprev)),torch.add(seasonal, lprev))

    def __smooth_seasonal(self, lprev,seasonal,error): ##done
        """Calculate the level"""

        seasonal = torch.add(self.seasonals[-self.seasonal_periods], torch.multiply(self.gamma, torch.multiply(torch.add(lprev,seasonal),error)))
        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))

    def fit(self):                 ##DONE

        """Fit the model to the data according to the equations:
     
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = self.initial_level
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
                
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])

                self.__smooth_error(row, self.level, seasonal)
                self.__smooth_level(self.level,seasonal,self.error)

                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))

                self.fitted[index] = torch.add(self.level,seasonal)
                
            
            else:

                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, seasonal)
                self.__smooth_seasonal(self.level,seasonal,self.error)
                self.__smooth_level(self.level,seasonal,self.error)

                self.fitted[index] = torch.add(self.level,seasonal)
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False): ###???????????????

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        if return_confidence:

            self.forecast =  torch.add(self.level,self.seasonals[-self.seasonal_periods]) #####?????????????
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):

                step_forecast = self.level + i * self.trend

                self.forecast = torch.cat((self.forecast, step_forecast))

                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

            return self.forecast, (upper_bounds, lower_bounds)
        

        else:

            self.forecast =  torch.add(self.level,self.seasonals[-self.seasonal_periods]) #####?????????????
            for i in range(1,h):
                self.forecast = torch.cat((self.forecast, self.level,self.trend))

            return self.forecast
