import torch
import numpy as np
from chronokit.base._models import Model
import scipy.optimize as opt
from scipy.linalg import toeplitz
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.preprocessing import differencing
import warnings
warnings.filterwarnings("ignore")

"""
ARIMA models for time series forecasting.
All methods have been implemented from chapter 8 of the textbook as a reference.
'Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
and practice. OTexts, 2014.'
"""

class ARIMA(Model):

    def __init__(self, data, p, d, q):
        """
        Non-Seasonal ARIMA model for Univariate time series data
        
        *Arguments:

        *p (int): Order of the autoregressive part
        *d (int): Order of differencing. Only order d=0 and d=1 are supported as of v1.0.x
        *q (int): Order of the moving average part
        """

        super().__init__(data)

        #Initialize orders
        self.p = p
        self.d = d
        self.q = q

        #Assertion for order values to be greater than 0 and type to be an int
        assert (self.p >= 0 and type(p) == int), "p must be an integer >= 0"
        assert (self.d >= 0 and type(d) == int), "d must be an integer >= 0"
        assert (self.q >= 0 and type(q) == int), "q must be an integer >= 0"

        #ARIMA(0,d,0) models are not implemented, as they correspond to just white noise or random walks with/without drift
        if self.p == 0 and self.q == 0:
            raise NotImplementedError("ARIMA models with p = 0 and q = 0 are not implemented")

        #Differencing of order grater than 1 for ARIMA models are not implemented as of v1.0.x
        if self.d > 1:
            raise NotImplementedError("ARIMA models with d > 1 are not implemented yet")
        
        #Get the ARIMA model type
        self.__model = {(False, False, True): "MA",
                        (False, True, True): "IMA",
                        (True, False, False): "AR",
                        (True, True, False): "ARI",
                        (True, False, True): "ARMA",
                        (True, True, True): "ARIMA"}[bool(self.p), bool(self.d), bool(self.q)]

        #Convert data to tensor an store it as orig_data
        self.orig_data = DataLoader(self.data).to_tensor()

        #Get the mean of the data
        self.c = torch.mean(self.orig_data, dtype=torch.float32)

        #If d > 0, difference the data
        if self.d != 0:
            self.data = differencing(self.orig_data.detach().clone(), order=d, return_numpy=False)
        else:
            #De-mean if d == 0
            self.data = self.orig_data.detach().clone() - self.c
        
        #Define self.fitted to assert whether the model was fitted before calling .predict()
        self.fitted = False

    def __yule_walker(self):
        """
        Yule-Walker equations for estimating phi parameters of AR(p)
        
        *Returns:
        
        *phi (array_like): Estimated phi parameters for AR(p)

        References:

        http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/YWSourceFiles/YW-Eshel.pdf
        """
        
        #Initialize r as 0s array to store autocorrelations
        r = torch.zeros(self.p+1)

        #Assign first index as the variance of the data
        r[0] = torch.sum(torch.square(self.data))/len(self.data)

        #Loop over length of order of AR(p) part
        for t in range(1, self.p+1):
            
            #Compute the each r_t by the autocovariance formula
            r[t] = torch.sum(self.data[:-t]*self.data[t:])/(len(self.data)-t)
        
        #Get the toeplitz matrix of autocovariances
        #R = [1, r1, r2, ..., rp-2, rp-1]
        #    [r1, 1, r2, ..., rp-3, rp-2]
        #     ...       ....         ...
        #    [rp-2, rp-3, rp-4, .., 1, 0]
        #    [rp-1, rp-2, rp-3, .. r1, 1]
        R = torch.tensor(toeplitz(r[:-1]), dtype=torch.float32)

        #R is full-rank and symmetric therefore invertible
        #phi = R^(-1)r
        phi = torch.matmul(torch.linalg.inv(R), r[1:])

        return phi

    def fit(self):
        """Fit the model to given data"""

        #Define sse(Sum of Squared Error) function for possibly estimating phi and theta if self.q != 0
        def sse(init_values):
            #Initialize phi and theta parameters from given values
            self.phi = torch.tensor(init_values[:self.p], dtype=torch.float64)
            self.theta = torch.tensor(init_values[self.p:], dtype=torch.float64)
            #Get the residuals from fitting with the parameters
            self.__get_resids()
            #Return the sum of squared errors
            err = np.array(self.errors)
            return np.sum(np.square(err))

        #Estimate phi parameters by yule-walker's
        #Will return empty tensor if self.p == 0
        self.phi = self.__yule_walker()

        #Initialize theta as None
        self.theta = None

        #Estimate theta parameters if the model has a MA part.
        if "MA" in self.__model:
            #Initialize init values to pass to sse function
            init_values = []
            
            #If the model has an AR part, use phi parameters that were estimated from yule-walker as initial values
            if "AR" in self.__model:
                init_values = list(self.phi)
            
            #Initialize theta parameters as 1s
            for i in range(self.q):
                init_values.append(torch.tensor(1))            

            #Get the estimated parameters by applying least squares on sse
            result_params = opt.least_squares(sse, init_values)
            result_params = list(result_params.x)

            #Assign estimated parameters
            if "AR" in self.__model:
                self.phi = torch.tensor(result_params[:self.p], dtype=torch.float32)
            self.theta = torch.tensor(result_params[self.p:], dtype=torch.float32)
        
        #Store model parameters
        self.params = {"c": self.c, "phi": self.phi, "theta": self.theta}

        #Get the residuals and fitted values with the estimated parameters
        self.__get_resids()

        #Assign fitted as True
        self.fitted = True
    
    def __get_resids(self):

        #Initialize errors and fitted values as a list
        self.errors = []
        self.fitted = []

        #Loop over the data
        for i in range(len(self.data)):
            
            #If the model is an AR model
            if self.__model in ["AR", "ARI"]:
                
                #Pass if not enough values for an AR equation 
                if i < self.p:
                    pass
                else:
                    y = self.data[i].detach().clone()

                    #Get the prediction by the AR equation
                    #AR(p) => y(hat)_t = phi(1)*y_t-1 + phi(2)*y_t-2 ... + phi(p)*y_t-p 
                    y_hat = torch.sum(self.phi*self.data[i-self.p:i].flip(0))

                    #Get the error and store it
                    err = y-y_hat
                    self.errors.append(err)
            
            #If the model is a MA model
            elif self.__model in ["MA", "IMA"]:

                y = self.data[i].detach().clone()

                #If there is enough errors observed;
                if self.q < len(self.errors):
                    #Get the prediction by the MA equation
                    #MA(q) => y(hat)_t = theta(1)*e_t-1 + theta(2)*e_t-2 ... + theta(p)*e_t-p 
                    y_hat = torch.sum(self.theta*torch.tensor(self.errors[-self.q:]).flip(0))

                    #Get the error and store it
                    err = y - y_hat
                    self.errors.append(err)
                #If not enough observed errors, add 0 to errors list
                else:
                    self.errors.append(torch.tensor(0))
            
            #If the model is an ARIMA model
            elif self.__model in ["ARMA", "ARIMA"]:

                #Pass if not enough values for an AR equation 
                if i < self.p:
                    pass
                else:
                    y = self.data[i].detach().clone()
                    #Get the prediction by the ARMA equation
                    #If not enough observed errors, predict by AR(p)
                    #ARMA(p,q) => y(hat)_t = phi(1)*y_t-1 + ... + phi(p)*y_t-p + theta(1)*error_t-1 + .... + theta(p)*error_t-p
                    y_hat = torch.sum(self.phi*self.data[i-self.p:i].flip(0))
                    if self.q < len(self.errors):
                        y_hat += torch.sum(self.theta*torch.tensor(self.errors[-self.q:]).flip(0))
                    
                    #Get the error and store it
                    err = y - y_hat
                    self.errors.append(err)
            
            #Add predicted values to the fitted array
            try:
                self.fitted.append(y_hat)
            except:
                self.fitted.append(torch.nan)
        
        #Convert fitted values to match with original data
        if self.d != 0:
            self.fitted = torch.tensor(self.fitted) + self.orig_data[-len(self.fitted):]
        else:
            self.fitted = torch.tensor(self.fitted) + self.c

        if len(self.fitted) != len(self.orig_data):
            start = torch.tensor([torch.nan for i in range(len(self.fitted), len(self.orig_data))])
            self.fitted = torch.cat((start, self.fitted))

    def predict(self, h):
        """
        Predict the next h values with the estimated parameters

        *Arguments:

        *h (int): Number of steps to predict

        Returns:

        *forecast (array_like): Predicted values
        """

        #Make necessary assertions
        if not self.fitted:
            raise Exception(".fit() must be called before predicting")
        
        assert (h > 0 and type(h) == int), "Provide h as an integer greater than 0"

        #Initialize forecasts as an empty list
        forecast = []

        ###TO DO###
        #Implement forecast equations suited to handling cases self.d > 1
        #Implement prediction intervals

        #Initialize data and possibly parameters to use for forecasting
        if self.__model == "ARI":
            #Use last p+d values of original data for forecasting
            fit_vals = self.orig_data[-self.p-self.d:]

            #If AR model with differencing change parameters to use for forecasting
            #Forecast equations for AR(p) will resemble AR(p+1) with parameters [(1+phi_1) - (phi_1-phi_2) - ... - (phi_p-1 - phi_p) - phi_p]
            phi_params = [self.phi[i] - self.phi[i+1] for i in range(len(self.phi)-1)]
            phi_params.insert(0, 1+self.phi[0])
            phi_params.append(self.phi[-1])
            phi_params = torch.tensor(phi_params)
        
        elif self.__model == "AR":
            #Use last p values of fitted data for forecasting
            fit_vals = self.data[-self.p:]

        elif self.__model in ["IMA", "MA"]:
            #Use the last q values of residuals for forecasting
            fit_errs = torch.tensor(self.errors[-self.q:])
        
        elif self.__model == "ARMA":
            #Use the last p values of fitted data and last q values of residuals for forecasting
            fit_vals = self.data[-self.p:]
            fit_errs = torch.tensor(self.errors[-self.q:])
        
        elif self.__model == "ARIMA":
            
            #Use the last p+d values of original data and last q values of residuals for forecasting
            fit_vals = self.orig_data[-self.p-self.d:].detach().clone()
            fit_errs = torch.tensor(self.errors[-self.q:])

            #Change parameters for the AR part to use for forecasting
            #Forecast equations for AR(p) will resemble AR(p+1) with parameters [(1+phi_1) - (phi_1-phi_2) - ... - (phi_p-1 - phi_p) - phi_p]
            phi_params = [self.phi[i] - self.phi[i+1] for i in range(len(self.phi)-1)]
            phi_params.insert(0, 1+self.phi[0])
            phi_params.append(self.phi[-1])
            phi_params = torch.tensor(phi_params)

        #Loop over forecast horizon
        for i in range(h):
            
            #Apply respective forecast equations for each model and store it in forecast array
            if self.__model == "ARI":

                res_arr = phi_params*fit_vals.flip(0)
                step_forecast = res_arr[0] - torch.sum(res_arr[1:])
            
            elif self.__model == "AR":
                step_forecast = torch.sum(self.phi*fit_vals.flip(0))
            
            elif self.__model in ["IMA", "MA"]:
                
                step_forecast = torch.sum(self.theta*fit_errs.flip(0))

                if self.__model == "MA":
                    step_forecast += self.c
            
            elif self.__model == "ARMA":
                step_forecast = torch.sum(self.phi*fit_vals.flip(0))
                step_forecast += torch.sum(self.theta*fit_errs.flip(0))
            
            elif self.__model == "ARIMA":

                res_arr = phi_params*fit_vals.flip(0)
                step_forecast = res_arr[0] - torch.sum(res_arr[1:])
                step_forecast += torch.sum(self.theta*fit_errs.flip(0))

            forecast.append(step_forecast)

            #Add the point forecast to the last index of the array used for forecasting the next step
            if self.__model in ["IMA", "MA"]:
                fit_errs = torch.cat((fit_errs[1:], torch.tensor([0.])))
            elif self.__model in ["AR", "ARI"]:
                fit_vals = torch.cat((fit_vals[1:], step_forecast.reshape(1)))
            else:
                fit_vals = torch.cat((fit_vals[1:], step_forecast.reshape(1)))
                fit_errs = torch.cat((fit_errs[1:], torch.tensor([0.])))

        forecast = torch.tensor(forecast, dtype=torch.float32)

        #Add the mean of data to the forecasts if AR(p) or ARMA(p,q) model
        if self.__model in ["AR", "ARMA"]:
            forecast += self.c
        
        #If MA(q) model with differencing, add the predicted values to the last value of the original data to get actual forecasts
        if self.__model == "IMA":

            forecast = self.orig_data[-1] + torch.cumsum(forecast, 0)

        return forecast