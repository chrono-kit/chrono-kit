import numpy as np
import pandas as pd
import torch
from exponential_smoothing.model import ETS_Model
from dataloader import DataLoader
import scipy.stats as stats

class ETS_MNM(ETS_Model):

    def __init__(self, dep_var,seasonal_periods=4, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        self.trend = None
        self.damped = False
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h):

        #no func
        return None

    def __smooth_level(self , lprev , error , seasonal):
        """Calculate the level"""

        self.level = torch.multiply(lprev, torch.add(1, torch.multiply(self.alpha, error)))

    def __smooth_error(self , y, lprev, error, seasonal):  

        """Calculate error"""
        y_hat = torch.multiply(seasonal,lprev)
        self.error = torch.divide(torch.sub(y, y_hat), y_hat)

    def __smooth_seasonal(self, seasonal,error,lprev):
        """Calculate the level"""

        seasonal = torch.multiply(seasonal, torch.add(1, torch.multiply(self.gamma, error)))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} * s_{t-m} * (1 + e_t)
        l_t = l_{t-1} * (1 + alpha * e_t)
        s_t = s_{t-m} * (1 + gamma*e_t)
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
                self.__smooth_level(self.level,self.error, seasonal)
                self.seasonals = torch.cat((self.seasonals, seasonal))

                self.fitted[index] = torch.multiply(self.level,seasonal)


            else:

                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, seasonal)
                self.__smooth_seasonal(seasonal,self.error, self.level)
                self.__smooth_level(self.level,self.error, seasonal)

                self.fitted[index] = torch.multiply(self.level,seasonal)

            self.__update_res_variance(self.error)

    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """
        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])
        
        len_s = self.seasonals.shape[0]

        for i in range(1,h+1):

            k = int((h-1)/self.seasonal_periods)
            
            step_forecast = torch.mul(self.level, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                
        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast

class ETS_ANM(ETS_Model):

    def __init__(self, dep_var,seasonal_periods=4, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        self.trend = None
        self.damped = False
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h):

        #no func
        return None

    def __smooth_level(self,lprev,error, seasonal):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.divide(torch.multiply(self.alpha, error), seasonal))
    
    def __smooth_error(self, y, lprev,seasonal):  

        """Calculate error"""

        y_hat = torch.multiply(seasonal,lprev)
        self.error = torch.sub(y,y_hat)

    def __smooth_seasonal(self, seasonal,error,lprev):
        """Calculate the level"""

        seasonal = torch.add(seasonal, torch.divide(torch.multiply(self.gamma, error),lprev))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} * s_{t-m}+ e_t
        l_t = l_{t-1} +alpha*e_t/s_{t-m}
        s_t = s_{t-m} + gamma*e_t/l_{t-1}
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
                self.__smooth_level(self.level,self.error, seasonal)
                self.seasonals = torch.cat((self.seasonals, seasonal))

                self.fitted[index] = torch.multiply(self.level,seasonal)
                
            
            else:

                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, seasonal)
                self.__smooth_seasonal(seasonal,self.error, self.level)
                self.__smooth_level(self.level,self.error, seasonal)

                self.fitted[index] = torch.multiply(self.level,seasonal)
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])
        
        len_s = self.seasonals.shape[0]

        for i in range(1,h+1):

            k = int((h-1)/self.seasonal_periods)
            
            step_forecast = torch.mul(self.level, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                
        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast

class ETS_AAN(ETS_Model):

    def __init__(self, dep_var, damped=False, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        self.trend = "add"
        self.damped = damped
        self.seasonal = None
        self.seasonal_periods = None
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h): 

        k = int((h-1)/self.seasonal_periods)
        
        if not self.damped:
            fc_var = torch.add(torch.square(self.alpha), torch.multiply(torch.multiply(self.alpha, self.beta), h))
            fc_var = torch.add(part1, torch.divide(torch.multiply(torch.square(self.beta), h*(2*h-1)), 6))
            fc_var = torch.multiply(h-1, part1)
            fc_var = torch.add(1, fc_var)
            fc_var = torch.multiply(self.residual_variance, fc_var)
        
        if self.damped:

            part1 = torch.multiply(torch.square(self.alpha), h-1)
            part2 = torch.multiply(torch.multiply(self.gamma, k), torch.add(torch.multiply(2, self.alpha), self.gamma))

            part3_1 = torch.multiply(torch.multiply(self.beta, self.phi), h)
            part3_1 = torch.divide(part3_1, torch.square(torch.subt(1, self.phi)))
            part3_2 = torch.add(torch.multiply(torch.multiply(2, self.alpha), torch.sub(1, self.phi)), torch.multiply(self.beta, self.phi))
            part3 = torch.multiply(part3_1, part3_2)

            part4_1 =  torch.multiply(torch.multiply(self.beta, self.phi), torch.sub(1, torch.pow(self.phi, h)))
            part4_1 = torch.divide(part4_1, torch.multiply(torch.square(torch.sub(1, self.phi)), torch.sub(1, torch.square(self.phi))))
            part4_2 = torch.multiply(torch.multiply(self.beta, self.phi), torch.add(1, torch.sub(torch.multiply(2, self.phi), torch.pow(self.phi, h))))
            part4_2 = torch.add(torch.multiply(torch.multiply(2, self.alpha), torch.sub(1, torch.square(self.phi))), part4_2)
            part4 = torch.multiply(part4_1, part4_2)

            fc_var = torch.add(part1, part2)
            fc_var = torch.add(fc_var, part3)
            fc_var = torch.sub(fc_var, part4)
            fc_var = torch.add(1, fc_var)
            fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, torch.sqrt(fc_var))
    
    def __smooth_level(self, lprev,bprev): ##done
        """Calculate the level"""

        self.level = torch.add(torch.add(lprev,torch.multiply(bprev, self.phi)), torch.multiply(self.alpha, self.error))
    
    def __smooth_error(self, y, lprev,bprev):  #done

        """Calculate error"""

        y_hat = torch.add(lprev, bprev)
        self.error = torch.sub(y, y_hat)

    def __smooth_trend(self, bprev): ##done
        """Calculate the level"""

        self.trend = torch.add(torch.multiply(bprev, self.phi), torch.multiply(self.beta, self.error))

    def fit(self):                 ##DONE

        """Fit the model to the data according to the equations:
        y_t = l_{t-1} + b_{t-1}+ e_t
        l_t = l_{t-1} + b_{t-1}+ alpha*e_t
        b_t = b_{t-1} + alpha*e_t
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = row[0]
                self.trend = row[0]
                self.error = torch.tensor(0, dtype=torch.float32)
                self.fitted[0] = row
            
            else:

                self.__smooth_error(row, self.level, self.trend)
                self.__smooth_level(self.level, self.trend)
                self.__smooth_trend(self.trend)
                self.fitted[index] = torch.add(self.level, self.trend)
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False): ###???????????????

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])

        for i in range(1,h+1):
            
            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

            step_forecast = torch.add(self.level, torch.multiply(damp_factor, self.trend))

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast

class ETS_MAN(ETS_Model):

    def __init__(self, dep_var, damped=False, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        
        self.trend = "add"
        self.damped = damped
        self.seasonal = None
        self.seasonal_periods = None
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h): 

        #no function
        return None

    def __smooth_level(self, lprev,bprev): #done
        """Calculate the level"""

        self.level = torch.multiply(torch.add(lprev,torch.multiply(self.phi, bprev)), torch.add(1, torch.multiply(self.alpha, self.error)))
    
    def __smooth_error(self, y, lprev,bprev):  #done

        """Calculate error"""

        y_hat = torch.add(lprev, torch.multiply(self.phi, bprev))
        self.error = torch.divide(torch.sub(y, y_hat), y_hat)
    
    def __smooth_trend(self, bprev,lprev): ##done

        """Calculate the level"""

        self.trend = torch.add(torch.multiply(self.phi, bprev), torch.multiply(self.beta, torch.multiply(torch.add(lprev,torch.multiply(self.phi, bprev)),self.error)))

    def fit(self):

        """Fit the model to the data according to the equations:
        y_t = (l_{t-1}+b_{t-1})*(1 + e_t)
        l_t = (l_{t-1}+b_{t-1})*(1 + alpha*e_t)
        b_t = b_{t-1}+beta*(l_{t-1}+b_{t-1})*e_t
        """

        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = row[0]
                self.trend=row[0]
                self.error = torch.tensor(0, dtype=torch.float32)
                self.fitted[0] = row
            
            else:

                self.__smooth_error(row, self.level)
                self.__smooth_level(self.level,self.trend)
                self.__smooth_trend(self.trend,self.level)
                self.fitted[index] = self.level
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
           This is again the same as the simple exponential smoothing forecast"""

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])

        for i in range(1,h+1):
            
            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

            step_forecast = torch.add(self.level, torch.multiply(damp_factor, self.trend))

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast


class ETS_MNA(ETS_Model):

    def __init__(self, dep_var,seasonal_periods=4, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        self.trend = None
        self.damped = False
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h): 

        #no function
        return None

    
    def __smooth_level(self,lprev,seasonal,error):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.alpha, torch.multiply(torch.add(lprev,seasonal),error)))
    
    def __smooth_error(self, y, lprev,seasonal):

        """Calculate error"""

        y_hat = torch.add(lprev,seasonal)

        self.error = torch.divide(torch.sub(y,y_hat),y_hat)

    def __smooth_seasonal(self, lprev,seasonal,error):
        """Calculate the level"""

        seasonal = torch.add(seasonal, torch.multiply(self.gamma, torch.multiply(torch.add(lprev,seasonal),error)))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def fit(self): 

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

                self.seasonals = torch.cat((self.seasonals, seasonal))

                self.fitted[index] = torch.add(self.level,seasonal)
                
            
            else:

                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, seasonal)
                self.__smooth_seasonal(self.level,seasonal,self.error)
                self.__smooth_level(self.level,seasonal,self.error)

                self.fitted[index] = torch.add(self.level,seasonal)
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])
        
        len_s = self.seasonals.shape[0]

        for i in range(1,h+1):

            k = int((h-1)/self.seasonal_periods)
            
            step_forecast = torch.add(self.level, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                
        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast                        
        
class ETS_ANA(ETS_Model):

    def __init__(self, dep_var,seasonal_periods=4, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        self.trend = None
        self.damped = False
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h):

        k = int((h-1)/self.seasonal_periods)

        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.add(fc_var, torch.multiply(torch.multiply(self.gamma, k), torch.add(torch.multiply(2, self.alpha), self.gamma)))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, fc_var)

    
    def __smooth_level(self,lprev,error): 
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.alpha, error))
    
    def __smooth_error(self, y, lprev, seasonal): 

        """Calculate error"""

        y_hat = torch.add(lprev, seasonal)
        self.error = torch.sub(y, y_hat)

    def __smooth_seasonal(self, seasonal, error):
        """Calculate the level"""

        seasonal = torch.add(seasonal, torch.multiply(self.gamma, error))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def fit(self): 
        """Fit the model to the data according to the equations:
        y_t = l_{t-1} + s_{t-m}+ e_t
        l_t = l_{t-1} +alpha*e_t
        s_t = s_{t-m} + gamma*e_t
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
                lprev = self.level
                self.__smooth_error(row, lprev, seasonal)
                self.__smooth_level(lprev,self.error)
                self.seasonals = torch.cat((self.seasonals, torch.tensor([seasonal])))

                self.fitted[index] = torch.add(self.level,seasonal)
                
            
            else:

                seasonal = self.seasonals[-self.seasonal_periods]
                lprev = self.level
                self.__smooth_error(row, lprev, seasonal)
                self.__smooth_level(lprev,self.error)
                self.__smooth_seasonal(seasonal,self.error)

                self.fitted[index] = torch.add(self.level,seasonal)
            
            self.__update_res_variance(self.error)
    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """
        
        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])
        
        len_s = self.seasonals.shape[0]

        for i in range(1,h+1):

            k = int((h-1)/self.seasonal_periods)
            
            step_forecast = torch.add(self.level, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                
        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast


class ETS_MAA(ETS_Model):
    def __init__(self, dep_var, seasonal_periods=4, damped=False, indep_var=None, **kwargs):

        """Implementation of Holt Winters Method with Damped Trend, Multiplicative Seasonality and Additive Errors"""
        self.trend = "add"
        self.damped = damped
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)

    def __get_confidence_interval(self, h):

        #NO EQUATION

        return None
    
    def __smooth_level(self, error, lprev,bprev,seasonal):
        """Calculate the level"""

        self.level = torch.multiply(self.alpha,torch.add(seasonal,torch.add(lprev,torch.multiply(self.phi, bprev))))
        self.level = torch.multiply(self.error,self.level)
        self.level = torch.add(self.level,torch.add(lprev,torch.multiply(self.phi, bprev)))
    
    def __smooth_error(self, y, lprev, bprev, seasonal):
        """Calculate error"""
        
        y_hat = torch.add(torch.add(lprev, torch.multiply(self.phi, bprev)), seasonal)

        self.error = torch.divide(torch.sub(y, y_hat), y_hat)

    def __smooth_seasonal(self, seasonal, error,lprev,bprev):
        seasonal = torch.add(seasonal, torch.multiply(error,torch.multiply(self.gamma,torch.add(seasonal,torch.add(lprev, torch.multiply(self.phi,bprev))))))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def __smooth_trend(self, error, lprev, bprev,seasonal):
        self.trend = torch.add(torch.multiply(self.phi, bprev), torch.multiply(error,torch.multiply(self.beta,torch.add(seasonal,torch.add(lprev,torch.multiply(self.phi,bprev))))))

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
                self.__smooth_level(self.error, lprev, bprev, seasonal)
                self.__smooth_trend(self.error, lprev, bprev, seasonal)
                self.seasonals = torch.cat((self.seasonals, seasonal))
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                self.__smooth_error(row, self.level, self.trend, seasonal)
                self.__smooth_seasonal(self.error)
                lprev, bprev = self.level, self.trend
                self.__smooth_level(self.error, lprev, bprev, seasonal)
                self.__smooth_trend(self.error, lprev, bprev, seasonal)
                self.fitted[index] = torch.add(self.level, torch.add(self.trend, seasonal))
            self.__update_res_variance(self.error)

    
    def predict(self, h, return_confidence=False):

        """Predict the next h-th value of the target variable
            Notice that the prediction equation is the same as simple
            exponential smoothing
        """

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])

        for i in range(1,h+1):
            
            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

            k = int((h-1)/self.seasonal_periods)
            len_s = self.seasonals.shape[0]

            step_forecast = torch.add(self.level, torch.multiply(damp_factor, self.trend))
            step_forecast = torch.add(step_forecast, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast

class ETS_AAA(ETS_Model):
    def __init__(self, dep_var, seasonal_periods=4, damped=False, indep_var=None, **kwargs):

        self.trend = "add"
        self.damped = damped
        self.seasonal = "add"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
    def __get_confidence_interval(self, h):

        k = int((h-1)/self.seasonal_periods)
        
        if not self.damped:
            part1 = torch.add(torch.square(self.alpha), torch.multiply(torch.multiply(self.alpha, self.beta), h))
            part1 = torch.add(part1, torch.divide(torch.multiply(torch.square(self.beta), h*(2*h-1)), 6))
            part1 = torch.multiply(h-1, part1)

            part2 = torch.add(torch.multiply(2,self.alpha), self.gamma)
            part2= torch.add(part2, torch.multiply(torch.multiply(self.beta, self.seasonal_periods), k+1))
            part2 = torch.multiply(part2, torch.multiply(self.gamma, k))

            fc_var = torch.add(part1, part2)
            fc_var = torch.add(1, fc_var)
            fc_var = torch.multiply(self.residual_variance, fc_var)
        
        if self.damped:

            part1 = torch.multiply(torch.square(self.alpha), h-1)
            part2 = torch.multiply(torch.multiply(self.gamma, k), torch.add(torch.multiply(2, self.alpha), self.gamma))

            part3_1 = torch.multiply(torch.multiply(self.beta, self.phi), h)
            part3_1 = torch.divide(part3_1, torch.square(torch.subt(1, self.phi)))
            part3_2 = torch.add(torch.multiply(torch.multiply(2, self.alpha), torch.sub(1, self.phi)), torch.multiply(self.beta, self.phi))
            part3 = torch.multiply(part3_1, part3_2)

            part4_1 =  torch.multiply(torch.multiply(self.beta, self.phi), torch.sub(1, torch.pow(self.phi, h)))
            part4_1 = torch.divide(part4_1, torch.multiply(torch.square(torch.sub(1, self.phi)), torch.sub(1, torch.square(self.phi))))
            part4_2 = torch.multiply(torch.multiply(self.beta, self.phi), torch.add(1, torch.sub(torch.multiply(2, self.phi), torch.pow(self.phi, h))))
            part4_2 = torch.add(torch.multiply(torch.multiply(2, self.alpha), torch.sub(1, torch.square(self.phi))), part4_2)
            part4 = torch.multiply(part4_1, part4_2)

            part5_1 = torch.multiply(torch.multiply(2, self.beta), torch.multiply(self.gamma, self.phi))
            part5_1 = torch.divide(part5_1, torch.multiply(torch.sub(1, self.phi), torch.sub(1, torch.pow(self.phi, self.seasonal_periods))))
            part5_2 = torch.multiply(torch.pow(self.phi, self.seasonal_periods), torch.sub(1, torch.pow(self.phi, torch.multiply(self.seasonal_periods, k))))
            part5_2 = torch.sub(torch.multiply(k, torch.sub(1, torch.pow(self.phi, self.seasonal_periods))), part5_2)
            part5 = torch.multiply(part5_1, part5_2)

            fc_var = torch.add(part1, part2)
            fc_var = torch.add(fc_var, part3)
            fc_var = torch.sub(fc_var, part4)
            fc_var = torch.add(fc_var, part5)
            fc_var = torch.add(1, fc_var)
            fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, torch.sqrt(fc_var))
    
    def __smooth_level(self, lprev,bprev):
        """Calculate the level"""

        self.level = torch.add(lprev, torch.multiply(self.phi, bprev))
        self.level = torch.add(self.level, torch.multiply(self.alpha, self.error))
    
    def __smooth_error(self, y, lprev,bprev,seasonal):

        """Calculate error"""

        y_hat = torch.add(lprev, torch.add(torch.multiply(self.phi, bprev), seasonal))

        self.error = torch.sub(y, y_hat)

    def __smooth_seasonal(self, seasonal, error):
        seasonal = torch.add(seasonal, torch.multiply(self.gamma, error))
        self.seasonals = torch.cat((self.seasonals, seasonal))

    def __smooth_trend(self, error, bprev):
        self.trend = torch.add(torch.multiply(error, self.beta), torch.multiply(self.phi, bprev))

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
                self.seasonals = torch.cat((self.seasonals, [seasonal]))
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

        upper_bounds = torch.tensor([])
        lower_bounds = torch.tensor([])

        for i in range(1,h+1):
            
            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

            k = int((h-1)/self.seasonal_periods)
            len_s = self.seasonals.shape[0]

            step_forecast = torch.add(self.level, torch.multiply(damp_factor, self.trend))
            step_forecast = torch.add(step_forecast, self.seasonals[len_s+i-self.seasonal_periods*k])

            self.forecast = torch.cat((self.forecast, step_forecast))

            if return_confidence:
                fc_var = self.__get_confidence_interval(h=i)
                upper_bounds = torch.cat((upper_bounds, torch.add(step_forecast, torch.abs(fc_var))))
                lower_bounds = torch.cat((lower_bounds, torch.sub(step_forecast, torch.abs(fc_var))))
                

        if return_confidence:
            return self.forecast, (upper_bounds, lower_bounds)
        else:
            return self.forecast

class ETS_ANN(ETS_Model):

    def __init__(self, dep_var, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Additive Errors"""
        
        self.trend = None
        self.damped = False
        self.seasonal = None
        self.seasonal_periods = None
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)

    def __get_confidence_interval(self, h):
        
        fc_var = torch.add(1 ,torch.multiply(torch.square(self.alpha),  h-1))
        fc_var = torch.multiply(self.residual_variance, fc_var)

        return torch.multiply(self.c, torch.sqrt(fc_var))
    
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
                self.level = self.initial_level
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

class ETS_AAM(ETS_Model):

    def __init__(self, dep_var, seasonal_periods=4, damped=False, **kwargs):

        """Implementation of Holt Winters Method with Damped Trend, Multiplicative Seasonality and Additive Errors"""

        self.trend = "add"
        self.damped = damped
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "add"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
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

        self.seasonals = torch.cat((self.seasonals, seasonal))
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

    
    def predict(self, h, confidence_level: float = None):

        """Predict the next h-th value of the target variable
           This is again the same as the simple exponential smoothing forecast"""

        if confidence_level:

            assert(type(confidence_level) == float and confidence_level < 1. and confidence_level > 0.), "Confidence level must be a float between 0 and 1"

            self.conf = confidence_level

            self.__calculate_conf_level()

            self.forecast = torch.multiply(self.seasonals[-self.seasonal_periods], torch.add(self.level, torch.multiply(self.phi, self.beta)))
            
            fc_var = self.__get_confidence_interval(h=0)
            upper_bounds = torch.add(self.forecast, torch.abs(fc_var))
            lower_bounds = torch.sub(self.forecast, torch.abs(fc_var))

            for i in range(1,h):

                damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i+1)]))

                k = torch.floor((i+1)/self.seasonal_periods)
                len_seasonal = self.seasonal.shape[0]
      
                step_forecast = torch.multiply(self.seasonals[len_seasonal+i-self.seasonal_periods*k], torch.add(self.level, torch.multiply(damp_factor, self.beta)))

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

class ETS_MNN(ETS_Model):

    def __init__(self, dep_var, indep_var=None, **kwargs):
        """Implementation of Simple Exponential Smoothing with Multiplicative Errors"""
        self.trend = None
        self.damped = False
        self.seasonal = None
        self.seasonal_periods = None
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        

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



class ETS_MAM(ETS_Model):

    def __init__(self, dep_var, seasonal_periods=4, damped=False, indep_var=None, **kwargs):

        self.trend = "add"
        self.damped = damped
        self.seasonal = "mul"
        self.seasonal_periods = seasonal_periods
        self.error_type = "mul"

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.error_type, self.seasonal_periods, self.damped, **kwargs)
        
        initialize_params = []
        for param in self.params:

            if param in kwargs:
                continue
            else:
                if param == "phi" and not self.damped:
                    continue
                else:
                    initialize_params.append(param)

        self.__initialize_params(initialize_params)

        self.residuals = torch.tensor([])
    
    def __calculate_conf_level(self):

        self.c = super().calculate_conf_level()

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)
    
    def __update_res_variance(self, error):

        self.residuals, self.residual_variance = super().update_res_variance(self.residuals, error)
        
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

        self.seasonals = torch.cat((self.seasonals, seasonal))
        
    
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
                seasonal = self.initial_seasonals[0]
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = self.initial_seasonals[index]
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