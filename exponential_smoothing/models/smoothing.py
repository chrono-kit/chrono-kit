import numpy as np
import torch
from exponential_smoothing.model import Smoothing_Model

class SES(Smoothing_Model):
    def __init__(self, dep_var, indep_var=None, **kwargs):
        self.trend = None
        self.damped = False
        self.seasonal = None
        self.seasonal_periods = None

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        
        super().__init__(dep_var, self.trend, self.seasonal, self.seasonal_periods, self.damped, **kwargs)
        
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

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)

    def __smooth_level(self, y, lprev):
        '''
        This function calculates the smoothed value for a given alpha and y value.
        Smoothing equation: 
        '''
        self.level =  torch.add(torch.mul(self.alpha, y),torch.mul((1 - self.alpha), lprev))
    
    def fit(self):
        '''
        This function fits the model to the data using smoothing equation.
        l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        '''
        self.fitted = torch.zeros(self.dep_var.shape[0])
        for index,row in enumerate(self.dep_var):
            if index == 0:
                self.level = row
                self.fitted[0] = row
            else:
                lprev = self.level
                self.__smooth_level(self.alpha, row, lprev)
                self.fitted[index] = self.level   
    
    def predict(self,h):
        '''
        This function predicts the next value in the series using the forecast equation.
        yhat_{t+h|t} = l_t
        '''
        self.forecast = torch.tensor([])
        for i in range(1,h+1):
            step_forecast = self.level
            self.forecast = torch.cat(self.forecast,step_forecast)
        return self.forecast


class HoltTrend(Smoothing_Model):
    def __init__(self, dep_var, damped=False, indep_var=None, **kwargs):
        self.trend = "add"
        self.damped = damped
        self.seasonal = None
        self.seasonal_periods = None

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.seasonal_periods, self.damped, **kwargs)
        
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


    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)

    def __smooth_level(self, y, lprev, bprev):
        '''This function calculates the smoothed level for a given alpha and y value'''
        self.level = torch.mul(torch.sub(1,self.alpha),torch.add(lprev, torch.mul(self.phi, bprev)))
        self.level = torch.add(torch.mul(self.alpha, y), self.level)

    def __smooth_trend(self, lprev, bprev):
        '''This function calculates the smoothed trend for a given beta and level values'''
        self.trend = torch.mul(self.beta, torch.sub(self.level, lprev))
        self.trend = torch.add(self.trend, torch.mul(torch.sub(1, self.beta), torch.mul(self.phi, bprev)))


    def fit(self):
        
        '''This function fits the model to the data using the following equations:
        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}'''
        
        self.fitted = np.zeros(self.dep_var.shape[0])
        for index, row in enumerate(self.dep_var):
            if index == 0:
                self.level = row[0]
                self.trend = 0
                self.fitted[0] = row[0]
            else:
                lprev, bprev = self.level, self.trend
                self.__smooth_level(row, lprev, bprev)
                self.__smooth_trend(lprev, bprev)
                self.fitted[index] = torch.add(self.level, torch.mul(self.phi, bprev))
        

    def predict(self, h):
        
        '''This function predicts the next h values in the series using the forecast equation.
        yhat_{t+h|t} = l_t + h * b_t'''
        
        self.forecast = torch.tensor([])
        for i in range(1,h+1):

            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i)]))

            step_forecast = torch.add(self.level, torch.mul(damp_factor, self.trend))
            self.forecast = torch.cat(self.forecast, step_forecast)
            
        return self.forecast
    
class HoltWinters(Smoothing_Model):
    def __init__(self, dep_var, seasonal="add", seasonal_periods=4, damped=False, indep_var=None, **kwargs):
        self.trend = "add"
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        super().set_allowed_kwargs(["alpha", "beta", "phi", "gamma"])
        super().__init__(dep_var, self.trend, self.seasonal, self.seasonal_periods, self.damped, **kwargs)
        
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

    def __initialize_params(self, initialize_params):

        super().initialize_params(initialize_params)

        self.initial_level = torch.tensor(self.init_components["level"])
        self.initial_trend = torch.tensor(self.init_components["trend"])
        self.initial_seasonals = torch.tensor(self.init_components["seasonal"])

        self.alpha, self.beta, self.gamma, self.phi = torch.tensor(list(self.params.values()), dtype=torch.float32)

    def __smooth_level(self, y, lprev, bprev, seasonal):
        '''This function calculates the smoothed level for a given alpha and y value'''

        if self.seasonal == "add":

            self.level = torch.mul(torch.sub(1,self.alpha),torch.add(lprev, torch.mul(self.phi, bprev)))
            self.level = torch.add(torch.mul(self.alpha, torch.sub(y, seasonal)), self.level)
        
        elif self.seasonal == "mul":

            self.level = torch.mul(torch.sub(1,self.alpha),torch.add(lprev, torch.mul(self.phi, bprev)))
            self.level = torch.add(torch.mul(self.alpha, torch.divide(y, seasonal)), self.level)


    def __smooth_trend(self, lprev, bprev):
        '''This function calculates the smoothed trend for a given beta and level values'''

        self.trend = torch.mul(self.beta, torch.sub(self.level, lprev))
        self.trend = torch.add(self.trend, torch.mul(torch.sub(1, self.beta), torch.mul(self.phi, torch.mul(self.phi, bprev))))

    def __smooth_seasonal(self,  y, lprev, bprev, seasonal):
        '''This function calculates the smoothed trend for a given beta and level values'''
        
        if self.seasonal == "add":

            seasonal = torch.mul(torch.sub(1,self.gamma), seasonal)
            seasonal = torch.add(seasonal, torch.mul(self.gamma, torch.sub(torch.sub(y, lprev), torch.mul(self.phi, bprev))))
            self.seasonals = torch.cat((self.seasonals, seasonal))

        elif self.seasonal == "mul":

            seasonal = torch.mul(torch.sub(1,self.gamma), seasonal)
            seasonal = torch.add(seasonal, torch.mul(self.gamma, torch.divide(y, torch.add(lprev,torch.mul(self.phi, bprev)))))
            self.seasonals = torch.cat((self.seasonals, seasonal))


    def fit(self):
        
        '''This function fits the model to the data using the following equations:
        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}'''
        
        self.fitted = np.zeros(self.dep_var.shape)

        for index, row in enumerate(self.dep_var):

            if index == 0:
                self.level = self.initial_level
                self.trend = self.initial_trend
                seasonal = self.initial_seasonals[0]
                self.seasonals = torch.tensor([seasonal])
                self.fitted[0] = row
            
            elif index < self.seasonal_periods:
                
                seasonal = self.initial_seasonals[index]
                lprev, bprev = self.level, self.trend
                self.__smooth_level(row, lprev, bprev, seasonal)
                self.__smooth_trend(lprev, bprev)

                self.seasonals = torch.cat((self.seasonals, seasonal))

                self.fitted[index] = torch.mul(seasonal, torch.add(self.level, torch.mul(self.phi, self.trend)))
                
            
            else:
                
                seasonal = self.seasonals[-self.seasonal_periods]
                lprev, bprev = self.level, self.trend
                self.__smooth_level(row, lprev, bprev, seasonal)
                self.__smooth_trend(lprev, bprev)
                self.__smooth_seasonal(row, lprev, bprev, seasonal)

                self.fitted[index] = torch.mul(seasonal, torch.add(self.level, torch.mul(self.phi, self.trend)))
            
            print(self.seasonals)

    def predict(self, h):
        
        '''This function predicts the next h values in the series using the forecast equation.
        yhat_{t+h|t} = l_t + h * b_t'''
        
        self.forecast = torch.tensor([])
        len_seasonal = self.seasonals.shape[0]
        for i in range(1,h+1):
       
            damp_factor = torch.sum(torch.tensor([torch.pow(self.phi, x+1) for x in range(i)]))
            k = torch.floor((i-1)/self.seasonal_periods)

            step_forecast = torch.add(self.level, torch.mul(damp_factor, self.trend))

            if self.seasonal == "mul":
                step_forecast = torch.add(step_forecast, self.seasonals[len_seasonal+i-self.seasonal_periods*(k+1)])
            elif self.seasonal == "add":
                step_forecast = torch.add(step_forecast, self.seasonals[len_seasonal+i-self.seasonal_periods*(k+1)])
            
            self.forecast = torch.cat(self.forecast, step_forecast)
            
        return self.forecast



            
