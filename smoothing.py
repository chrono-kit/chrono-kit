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
            