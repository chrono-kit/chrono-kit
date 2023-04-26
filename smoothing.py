import numpy as np
import pandas as pd
from model import Model

class SES(Model):
    def __init__(self, dep_var, indep_var=None, **kwargs):
        super().set_allowed_kwargs(['alpha'])
        super().__init__(dep_var, indep_var, **kwargs)

    def __smooth(self, alpha, y, lprev):
        '''
        This function calculates the smoothed value for a given alpha and y value
        '''
        return alpha * y + (1 - alpha) * lprev

    def fit(self):
        '''
        This function fits the model to the data using smoothing equation
        l_t = alpha * y_t + (1 - alpha) * l_{t-1}
        '''
        self.fitted = np.zeros(self.dep_var.shape[0])
        for index, row in self.dep_var.iterrows():
            if index == 0:
                self.smooth = row[0]
                self.fitted[0] = row[0]
            else:
                self.smooth = self.__smooth(self.alpha, row[0], self.smooth)
                self.fitted[index] = self.smooth
        return self.smooth

    def predict(self,h):
        '''
        This function predicts the next value in the series using the forecast equation.
        yhat_{t+h|t} = l_t
        '''
        self.forecast =np.array([self.smooth])
        for i in range(1,h):
            self.forecast = np.append(self.forecast,self.__smooth(self.alpha, self.forecast[i-1], self.forecast[i-1]))
        return self.forecast
    



class HoltTrend(Model):
    def __init__(self, dep_var, indep_var=None, **kwargs):
        super().set_allowed_kwargs(['alpha', 'beta', 'damped'])
        super().__init__(dep_var, indep_var, **kwargs)

    def __smooth_level(self, alpha, y, lprev, bprev):
        '''This function calculates the smoothed level for a given alpha and y value'''
        return alpha * y + (1 - alpha) * (lprev + bprev)

    def __smooth_trend(self, beta, lcurr, lprev, bprev):
        '''This function calculates the smoothed trend for a given beta and level values'''
        return beta * (lcurr - lprev) + (1 - beta) * bprev

    def fit(self):
        
        '''This function fits the model to the data using the following equations:
        l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}'''
        
        self.fitted = np.zeros(self.dep_var.shape[0])
        for index, row in self.dep_var.iterrows():
            if index == 0:
                self.level = row[0]
                self.trend = 0
                self.fitted[0] = row[0]
            else:
                self.level = self.__smooth_level(self.alpha, row[0], self.level, self.trend)
                self.trend = self.__smooth_trend(self.beta, self.level, self.fitted[index-1], self.trend)
                self.fitted[index] = self.level + self.trend
                if self.damped:
                    self.trend *= self.beta_damp
        return self.level, self.trend

    def predict(self, h):
        
        '''This function predicts the next h values in the series using the forecast equation.
        yhat_{t+h|t} = l_t + h * b_t'''
        
        self.forecast = np.zeros(h)
        for i in range(h):
            self.forecast[i] = self.level + (i+1) * self.trend
            if self.damped:
                self.forecast[i] *= self.beta_damp ** (i+1)
        return self.forecast


class HoltWinters:
    def __init__(self, data, seasonal_periods=4, alpha=0.2, beta=0.2, gamma=0.2, forecast_periods=4):
        self.data = data
        self.seasonal_periods = seasonal_periods
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.forecast_periods = forecast_periods
        
    def holt_winters_additive(self):
        """
        Holt-Winters additive seasonal method for time series forecasting.

        Returns:
        forecast (pandas.DataFrame): DataFrame containing the forecasted values and their confidence intervals.
        """

        # Split data into training and validation sets
        train = self.data[:-self.forecast_periods]
        val = self.data[-self.forecast_periods:]

        # Initialize parameters
        level = np.mean(train)
        trend = np.mean(np.diff(train))
        seasonals = np.zeros(self.seasonal_periods)

        # Initialize forecast
        forecast = np.zeros(len(train) + self.forecast_periods)

        # Calculate initial seasonals
        for i in range(self.seasonal_periods):
            seasonals[i] = np.mean(train[i::self.seasonal_periods] - level)

        # Perform forecasting
        for i in range(len(train) + self.forecast_periods):
            if i < len(train):
                # Update level
                level = self.alpha * (train[i] - seasonals[i % self.seasonal_periods]) + (1 - self.alpha) * (level + trend)
                # Update trend
                trend = self.beta * (level - level[i - 1]) + (1 - self.beta) * trend
                # Update seasonals
                seasonals[i % self.seasonal_periods] = self.gamma * (train[i] - level) + (1 - self.gamma) * seasonals[i % self.seasonal_periods]
            else:
                # Forecast
                forecast[i] = level + (i - len(train) + 1) * trend + seasonals[i % self.seasonal_periods]

        # Construct forecast DataFrame
        forecast_df = pd.DataFrame(forecast[-self.forecast_periods:], index=val.index, columns=["Forecast"])

        return forecast_df
    
    def holt_winters_multiplicative(self):
        """
        Holt-Winters multiplicative seasonal method for time series forecasting.

        Returns:
        forecast (pandas.DataFrame): DataFrame containing the forecasted values and their confidence intervals.
        """

        # Split data into training and validation sets
        train = self.data[:-self.forecast_periods]
        val = self.data[-self.forecast_periods:]

        # Initialize parameters
        level = np.mean(train)
        trend = np.mean(np.diff(train))
        seasonals = np.ones(self.seasonal_periods)

        # Initialize forecast
        forecast = np.zeros(len(train) + self.forecast_periods)

        # Calculate initial seasonals
        for i in range(self.seasonal_periods):
            seasonals[i] = np.mean(train[i::self.seasonal_periods] / level)

        # Perform forecasting
        for i in range(len(train) + self.forecast_periods):
            if i < len(train):
                # Update level
                level = self.alpha * (train[i] / seasonals[i % self.seasonal_periods]) + (1 - self.alpha) * (level * trend)
                # Update trend
                trend = self.beta * (level / level[i - 1]) + (1 - self.beta) * trend
                # Update seasonals
                seasonals[i % self.seasonal_periods] = self.gamma * (train[i] / level) + (1 - self.gamma) * seasonals[i % self.seasonal_periods]
        else:
            # Forecast
            forecast[i] = (level + (i - len(train) + 1) * trend) * seasonals[i % self.seasonal_periods]

        # Construct forecast DataFrame
        forecast_df = pd.DataFrame(forecast[-self.forecast_periods:], index=val.index, columns=["Forecast"])

        return forecast_df
            
