import numpy as np
import pandas as pd
import torch
from model import ETS_Model
from dataloader import DataLoader
import scipy.stats as stats
from scipy.optimize import least_squares
from initialization_methods import heuristic_initialization

class ETS_ANN(ETS_Model):

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


class ETS_MNN(ETS_Model):

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


class ETS_AAdM(ETS_Model):

    def __init__(self, dep_var, seasonal_periods=4, indep_var=None, **kwargs):

        """Implementation of Holt Winters Method with Damped Trend, Multiplicative Seasonality and Additive Errors"""

        self.trend = "add"
        self.damped = True
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma", "conf"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, 
                            indep_var, **kwargs)
        

        if "alpha" not in kwargs:

            self.__initialize(components_only=False)
        
        else:

            self.__initialize(components_only=True)
            
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
    
    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, trend, seasonal = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_trend = trend
        self.initial_seasonals = seasonal

        alpha = 0.1
        beta = 0.01
        gamma = 0.01
        phi = 0.99
        
        self.params = self.__estimate_params(
                        init_components = np.array([self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods]),
                        params = np.array([alpha, beta, gamma, phi]))
        
        self.alpha, self.beta, self.gamma, self.phi = self.params

    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()
    
    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        

    def __get_confidence_interval(self, h):

        
        ###No known equation for this method, will implement later
        pass
    
    def __smooth_level(self, error, lprev, bprev, seasonal):
        """Calculate the level"""

        self.level = torch.add(torch.multiply(self.phi, bprev), torch.divide(torch.multiply(self.alpha, error), seasonal))
        self.level = torch.add(self.level, lprev)
    
    def __smooth_trend(self, error, bprev, seasonal):

        self.trend = torch.divide(torch.multiply(self.beta, error), seasonal)
        self.trend = torch.add(self.trend, torch.multiply(self.phi, bprev))

    def __smooth_seasonal(self, error, lprev, bprev):

        seasonal = torch.divide(torch.multiply(self.gamma, error), torch.add(lprev, torch.multiply(self.phi, bprev)))
        seasonal = torch.add(seasonal, self.seasonals[-self.seasonal_periods])

        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))
        #print(self.seasonals)
        
    
    def __smooth_error(self, y, lprev, bprev, seasonal):

        """Calculate error"""
        
        y_hat = torch.multiply(torch.add(lprev, torch.multiply(self.phi, bprev)), seasonal)

        self.error = torch.sub(y, y_hat)


    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1}*(1 + e_t)
        l_t = l_{t-1}*(1 + alpha*e_t)
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_level(self.error, self.level, self.trend, seasonal)
                self.__smooth_trend(self.error, self.trend, seasonal)

                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))

                self.fitted[index] = torch.multiply(seasonal, torch.add(self.level, torch.multiply(self.phi, self.trend)))
                
            
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error, self.level, self.trend)
                self.__smooth_level(self.error, self.level, self.trend, seasonal)
                self.__smooth_trend(self.error, self.trend, seasonal)


                self.fitted[index] = torch.multiply(seasonal, torch.add(self.level, torch.multiply(self.phi, self.trend)))

            
            self.__update_res_variance(self.error)

    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
           This is again the same as the simple exponential smoothing forecast"""

        if return_confidence:

            self.forecast = torch.multiply(self.seasonals[-self.seasonal_periods], torch.add(self.level, torch.multiply(self.phi, self.beta)))
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):

                damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

                k = torch.floor((i+1)/self.seasonal_periods)
      
                step_forecast = torch.multiply(self.seasonals[i+1-self.seasonal_periods*k], torch.add(self.level, torch.multiply(damp_factor, self.beta)))

                self.forecast = torch.cat((self.forecast, step_forecast))

                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

            return self.forecast, (upper_bounds, lower_bounds)
        

        else:
            
            self.forecast = torch.multiply(self.seasonals[-self.seasonal_periods], torch.add(self.level, torch.multiply(self.phi, self.trend)))

            for i in range(1,h):
                damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

                k = int(torch.floor(torch.tensor((i+1)/self.seasonal_periods)).numpy())
      
                step_forecast = torch.multiply(self.seasonals[i+1-self.seasonal_periods*(k+1)], torch.add(self.level, torch.multiply(damp_factor, self.trend)))

                self.forecast = torch.cat((self.forecast, step_forecast))

            return self.forecast


class ETS_MAdM(ETS_Model):

    def __init__(self, dep_var, seasonal_periods=4, indep_var=None, **kwargs):

        """Implementation of Holt Winters Method with Damped Trend, Multiplicative Seasonality and Additive Errors"""
        self.trend = "add"
        self.damped = True
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma", "conf"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, 
                            indep_var, **kwargs)
        

        if "alpha" not in kwargs:

            self.__initialize(components_only=False)
        
        else:

            self.__initialize(components_only=True)
            
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
    
    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, trend, seasonal = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_trend = trend
        self.initial_seasonals = seasonal

        alpha = 0.1
        beta = 0.01
        gamma = 0.01
        phi = 0.99
        
        self.params = self.__estimate_params(
                        init_components = np.array([self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods]),
                        params = np.array([alpha, beta, gamma, phi]))
        
        self.alpha, self.beta, self.gamma, self.phi = self.params


    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()
    
    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        

    def __get_confidence_interval(self, h):

        
        ###No known equation for this method, will implement later
        pass
    
    def __smooth_level(self, error, lprev, bprev):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.phi, bprev))
        self.level = torch.multiply(self.level, torch.add(1, torch.multiply(self.alpha, error)))
    
    def __smooth_trend(self, error, lprev, bprev):

        self.trend = torch.multiply(torch.multiply(error, self.beta), torch.add(lprev , torch.multiply(self.phi, bprev)))
        self.trend = torch.add(self.trend, torch.multiply(self.phi, bprev))

    def __smooth_seasonal(self, error):

        seasonal = torch.multiply(self.seasonal_periods, torch.add(1, torch.multiply(self.gamma, error)))

        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))
        
    
    def __smooth_error(self, y, lprev, bprev, seasonal):

        """Calculate error"""
        
        y_hat = torch.multiply(torch.add(lprev, torch.multiply(self.phi, bprev)), seasonal)

        self.error = torch.divide(torch.sub(y, y_hat), y_hat)


    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1}*(1 + e_t)
        l_t = l_{t-1}*(1 + alpha*e_t)
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])
                self.__smooth_error(row, self.level, self.trend, seasonal)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)

                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))

                self.fitted[index] = torch.multiply(seasonal, torch.add(self.level, torch.multiply(self.phi, self.trend)))
                
            
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)

                self.fitted[index] = torch.multiply(seasonal, torch.add(self.level, torch.multiply(self.phi, self.trend)))

            
            self.__update_res_variance(self.error)

    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
           This is again the same as the simple exponential smoothing forecast"""

        if return_confidence:

            self.forecast = torch.multiply(self.seasonals[-self.seasonal_periods], torch.add(self.level, torch.multiply(self.phi, self.beta)))
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):

                damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

                k = torch.floor((i+1)/self.seasonal_periods)
      
                step_forecast = torch.multiply(self.seasonals[i+1-self.seasonal_periods*k], torch.add(self.level, torch.multiply(damp_factor, self.beta)))

                self.forecast = torch.cat((self.forecast, step_forecast))

                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

            return self.forecast, (upper_bounds, lower_bounds)
        

        else:
            
            self.forecast = torch.multiply(self.seasonals[-self.seasonal_periods], torch.add(self.level, torch.multiply(self.phi, self.trend)))

            for i in range(1,h):
                damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

                k = int(torch.floor(torch.tensor((i+1)/self.seasonal_periods)).numpy())
      
                step_forecast = torch.multiply(self.seasonals[i+1-self.seasonal_periods*(k+1)], torch.add(self.level, torch.multiply(damp_factor, self.trend)))

                self.forecast = torch.cat((self.forecast, step_forecast))

            return self.forecast
class ETS_AAA(ETS_Model):
    def __init__(self, dep_var, seasonal_periods=4, indep_var=None, **kwargs):
        self.trend = "add"
        self.damped = False
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "gamma", "conf"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, 
                            indep_var, **kwargs)
        

        if "alpha" not in kwargs:

            self.__initialize(components_only=False)
        
        else:

            self.__initialize(components_only=True)
            
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
    
    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, trend, seasonal = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_trend = trend
        self.initial_seasonals = seasonal

        alpha = 0.1
        beta = 0.01
        gamma = 0.01
        
        self.params = self.__estimate_params(
                        init_components = np.array([self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods]),
                        params = np.array([alpha, beta, gamma]))
        
        self.alpha, self.beta, self.gamma= self.params

    
    def __calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""
        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        


    def __get_confidence_interval(self, h):

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self, lprev,bprev):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.alpha, self.error))
        self.level = torch.add(self.level, bprev)
    
    def __smooth_error(self, y, lprev,bprev,seasonal):

        """Calculate error"""

        self.error = torch.sub(y, lprev)
        self.error = torch.sub(self.error, bprev)
        self.error = torch.sub(self.error, seasonal)
        
    def __smooth_seasonal(self, error):
        seasonal = torch.add(self.seasonal_periods, torch.multiply(self.gamma, error))
        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))

    def __smooth_trend(self, error, bprev):
        self.trend = torch.add(torch.multiply(error, self.beta),bprev)
        self.trend = torch.add(self.trend, torch.multiply(self.phi, bprev))

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} + b_{t-1} + s_{t-m}+ e_t
        l_t = l_{t-1} + b_{t-1} + alpha*e_t
        """
        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):
            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])
                self.__smooth_error(row, self.level, self.trend, seasonal)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)
                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
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
        
class ETS_AAM(ETS_Model):
    def __init__(self, dep_var, seasonal_periods=4, indep_var=None, **kwargs):
        self.trend = "add"
        self.damped = False
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "gamma", "conf"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, 
                            indep_var, **kwargs)
        

        if "alpha" not in kwargs:
            self.__initialize(components_only=False)
        
        else:

            self.__initialize(components_only=True)
            
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
    
    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, trend, seasonal = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_trend = trend
        self.initial_seasonals = seasonal

        alpha = 0.1
        beta = 0.01
        gamma = 0.01
        
        self.params = self.__estimate_params(
                        init_components = np.array([self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods]),
                        params = np.array([alpha, beta, gamma]))
        
        self.alpha, self.beta, self.gamma= self.params

    
    def __calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""
        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        


    def __get_confidence_interval(self, h):

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self, lprev,bprev,seasonal):
        """Calculate the level"""

        self.level = torch.mul(seasonal, torch.add(lprev, bprev))
    
    def __smooth_error(self, y, lprev,bprev,seasonal):
        """Calculate error"""

        self.error = torch.sub(y, torch.mul(seasonal, torch.add(lprev, bprev)))
        
    def __smooth_seasonal(self, error,lprev,bprev):
        seasonal = torch.add(self.seasonal_periods, torch.divide(torch.multiply(self.gamma, error),torch.add(lprev,bprev)))
        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))

    def __smooth_trend(self, error, bprev,seasonal):
        self.trend = torch.add(bprev, torch.divide(torch.multiply(self.beta, error),seasonal))

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} + b_{t-1} + s_{t-m}+ e_t
        l_t = l_{t-1} + b_{t-1} + alpha*e_t
        """
        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):
            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])
                self.__smooth_error(row, self.level, self.trend, seasonal)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error,lprev,bprev,seasonal)
                self.__smooth_trend(self.error,bprev,seasonal)
                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))
                self.fitted[index] = torch.mul(seasonal,torch.add(lprev,bprev))
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error,self.level, self.trend)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error,lprev,bprev,seasonal)
                self.__smooth_trend(self.error,bprev,seasonal)
                self.fitted[index] = torch.mul(seasonal,torch.add(lprev,bprev))
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
class ETS_MAA(ETS_Model):
    def __init__(self, dep_var, seasonal_periods=4, indep_var=None, **kwargs):

        """Implementation of Holt Winters Method with Damped Trend, Multiplicative Seasonality and Additive Errors"""
        self.trend = "add"
        self.damped = False
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha","beta","gamma","conf"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, 
                            indep_var, **kwargs)
        
        if "alpha" not in kwargs:
            self.__initialize(components_only=False)
        
        else:

            self.__initialize(components_only=True)
            
        if "conf" not in kwargs:
            self.conf = 0.95
        
        self.__calculate_conf_level()

        self.residuals = torch.tensor([])
    
    def __estimate_params(self, init_components, params):
        return super().estimate_params(init_components, params)

    def __initialize(self, components_only=False):

        lvl, trend, seasonal = heuristic_initialization(self.dep_var, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods)

        self.initial_level = lvl
        self.initial_trend = trend
        self.initial_seasonals = seasonal

        alpha = 0.1
        beta = 0.01
        gamma = 0.01
        
        self.params = self.__estimate_params(
                        init_components = np.array([self.initial_level, self.initial_trend, self.initial_seasonals, self.seasonal_periods]),
                        params = np.array([alpha, beta, gamma]))
        
        self.alpha, self.beta, self.gamma = self.params

    
    def __calculate_conf_level(self):

        """Calculate the confidence level to be used for intervals"""
        self.c = round(stats.norm.ppf(1 - ((1 - self.conf) / 2)), 2)

    def __update_res_variance(self, error):

        self.residuals = torch.cat((self.residuals, torch.reshape(error, (1,1))))
        

        res_mean = torch.sum(self.residuals)/self.residuals.shape[0]

        self.residual_variance = torch.sum(torch.square(torch.sub(self.residuals, res_mean)))
        self.residual_variance = torch.divide(self.residual_variance, self.residuals.shape[0]-1)
        


    def __get_confidence_interval(self, h):

        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self, lprev,bprev,seasonal):
        """Calculate the level"""

        self.level = torch.add(lprev,bprev)
        self.level = torch.add(self.level,torch.multiply(self.error,torch.multiply(self.alpha,torch.add(seasonal,torch.add(lprev,bprev)))))
    
    def __smooth_error(self, y, lprev, bprev, seasonal):
        """Calculate error"""
        
        y_hat = torch.add(torch.add(lprev, bprev), seasonal)

        self.error = torch.divide(torch.sub(y, y_hat), y_hat)

    def __smooth_seasonal(self, error,lprev,bprev):
        seasonal = torch.add(self.seasonal_periods, torch.multiply(error,torch.multiply(self.gamma,torch.add(self.seasonal_periods,torch.add(lprev,bprev)))))
        self.seasonals = torch.cat((self.seasonals, torch.tensor(seasonal)))

    def __smooth_trend(self, error, lprev, bprev,seasonal):
        self.trend = torch.add(self.bprev, torch.multiply(error,torch.multiply(self.gamma,torch.add(seasonal,torch.add(lprev,bprev)))))


    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = (l_{t-1} + b_{t-1} + s_{t-m})(1+e_t)
        l_t = l_{t-1} + b_{t-1} + alpha*(l_{t-1} + b_{t-1} + s_{t-m})e_t
        """
        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):
            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                self.error = torch.tensor(0, dtype=torch.float32)
                seasonal = torch.tensor(self.initial_seasonals[0])
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = torch.tensor(self.initial_seasonals[index])
                self.__smooth_error(row, self.level, self.trend, seasonal)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)
                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev)
                self.__smooth_trend(self.error, lprev, bprev)
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
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