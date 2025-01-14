import torch
import numpy as np
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.base._models import TraditionalTimeSeriesModel
from chronokit.arima.__initialization import SARIMAInitializer

class _B_Op_:
    """
    Stands for Backshift operator

    _B_Op_(weight=1, order=1) * y_t = y_(t-1)
    _B_Op_(weight=2, order=3) * y_t = 2*y_(t-3)

    _B_Op_(weight=x, order=d) * _B_Op_(weight=y, order=k) = _B_Op_(weight=x*y, order=d+k)

    """
    def __init__(self, weight=1, order=1):
        
        self.order = order
        self.weight = weight

    def __mul__(self, other):

        if not isinstance(other, type(self)):
            try:
                iter(other)
                return self.weight*other[-1 - self.order]

            except:  # noqa: E722
                return _B_Op_(weight=self.weight*other,
                            order=self.order)
        else:
            return _B_Op_(weight=self.weight*other.weight,
                       order=self.order+other.order)

    def __rmul__(self, other):

        return self.__mul__(other)
    

class _Dist_Brackets_:
    """
    Special Brackets type for utilizing _B_Op_ class
    Basically, multiplying by using this class 
    multiplies the parameters inside the brackets by distributive law
    """
    def __init__(self, *args):

        self.args = list(args)

        if len(self.args) == 1 and isinstance(self.args[0], tuple):
            self.args = list(self.args[0])
      
    def __mul__(self, other):

        if isinstance(other, type(self)):
            args_list = []
            for arg in self.args:
                for arg2 in other.args:
                    args_list.append(arg*arg2)
            
            return _Dist_Brackets_(tuple(args_list))

        else:
            #Then other is either y_t array ot e_t array
            return_val = 0
            
            for arg in self.args:
                if isinstance(arg, type(_B_Op_())):
                    return_val += arg*other
                
                #this arg is a weight argument (1, phi, theta etc...)
                else:
                    return_val += arg*other[-1]
        
            return DataLoader(return_val).to_tensor()

    def __rmul__(self, other):

        return self.__mul__(other)

class _Diff_:

    def __new__(self, ord=1, seasonal_period=None):
        """
        Differencing operator used in ARIMA models
        Differencing of order 1 = (1 - B)*y_t
        Differencing of order d = ((1 - B)**d)*y_t
        """
        if ord == 0:
            return _Dist_Brackets_(1)
        
        m = 1 if seasonal_period is None else seasonal_period
 
        op = _Dist_Brackets_(1, -1*_B_Op_(order=m))

        for x in range(1, ord):
            op = op*_Dist_Brackets_(1, -1*_B_Op_(order=m))

        return op

class ARIMAProcess(TraditionalTimeSeriesModel):

    def __init__(self, data, initialization_method, **kwargs):
        """
        Base class for all models based on ARIMA Processes.
        This class handles initialization for parameters and attributes
        of different specialized ARIMA based models
    
        Autoregressive process, AR(p) : (1 - phi_1*B - ... - phi_p*(B**p))*y_t = e_t

        Moving Average process, MA(q): y_t = (1 - theta_1*B - ... - theta_q*(B**q))*e_t

        Mixed Autoregressive-Moving Average Integrated Process, ARIMA(p,d,q):

        (1 - phi_1*B - ... - phi_p*(B**p))((1-B)**d)*y_t = (1 - theta_1*B - ... - theta_q*(B**q))*e_t

        All ARIMA models are developed by the below book as a reference;

        'Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
        Time series analysis: Forecasting and control (5th ed).'

        This book may also be referenced as 'Box, G. et al." throughout this repository
        """
        
        super().__init__(data, **kwargs)

        #Check model validity
        self.__check_arima()
        self.__init_model_info()

        if initialization_method != "known":
            
            initializer = SARIMAInitializer(self, initialization_method)

            init_params = initializer.initialize_parameters()

            self.phi = init_params["phi"]
            self.theta = init_params["theta"]
            self.seasonal_phi = init_params["seasonal_phi"]
            self.seasonal_theta = init_params["seasonal_theta"]

        else:
            if not set(self.allowed_kwargs).issubset(kwargs.keys()):
                raise Exception(
                    f"All values component should be passes when \
                    initialization_method='known'\n {self.allowed_kwargs}"
                )
        
        self.set_kwargs(kwargs)
            
        self.params = {"phi": self.phi,
                       "theta": self.theta,
                       "seasonal_phi": self.seasonal_phi,
                       "seasonal_theta": self.seasonal_theta}
        
        self.__prepare_operations__()

        self.phi = DataLoader(self.phi).to_tensor()
        self.theta = DataLoader(self.theta).to_tensor()
        self.seasonal_phi = DataLoader(self.seasonal_phi).to_tensor()
        self.seasonal_theta = DataLoader(self.seasonal_theta).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        self.errors = torch.zeros(size=self.data.shape)
    
    def calc_iv_weights(self, phi, theta):

        pi = [-1]
        for i in range(20):
            p = 0
            cur_pi = np.array(pi).copy()
            if len(pi) < len(theta):
                cur_pi = np.concatenate((np.zeros(len(theta) - len(pi)), cur_pi))
            cur_theta = theta.copy()
            if len(pi) > len(cur_theta):
                cur_theta = np.concatenate((cur_theta, np.zeros(len(pi) - len(cur_theta))))
            
            p += np.sum(cur_theta[-1::-1]*cur_pi)
            if i < len(phi):
                p += phi[i]
            if abs(p) < 1e-7 and i >= len(phi) + len(theta):
                break
            pi.append(p)

        return np.array(pi[1:])
    
    def stationarity_condition(self, ar_weights):
        poly_roots = np.roots(np.concatenate(([1], -ar_weights)))
        
        if max(abs(poly_roots)) > 1:
            return False
        
        return True
    
    def invertibility_condition(self, ma_weights):
        poly_roots = np.roots(np.concatenate((-ma_weights[-1::-1], [1])))
        
        if min(abs(poly_roots)) < 1:
            return False
        
        return True
    
    def __prepare_operations__(self):
        """
        Prepare operations with model parameters

        _B_Op_(weight=-phi[x], order=x+1) = -phi_x*(B**(x+1))

        _Dist_Brackets_(1, 
                        _B_Op_(weight=-phi[0], order=1), 
                        ..., 
                        _B_Op_(weight=-phi[p-1], order=p)
        )

                        = (1 - phi_1*B - .... - phi_p*(B**p))

        _Diff_(ord=d, seasonal_periods=None) = (1 - B)**d
        _Diff_(ord=d, seasonal_periods=m) = (1 - B**m)**d    
        """
        
        diff_operator = _Diff_(ord=self.d, seasonal_period=None)
        seasonal_diff_operator = _Diff_(ord=self.D, seasonal_period=self.seasonal_periods)

        phi = DataLoader(self.phi).to_numpy()
        theta = DataLoader(self.theta).to_numpy()

        seasonal_phi = DataLoader(self.seasonal_phi).to_numpy()
        seasonal_theta = DataLoader(self.seasonal_theta).to_numpy()

        iv_weights = self.calc_iv_weights(phi, theta)
        seasonal_iv_weights = self.calc_iv_weights(seasonal_phi, seasonal_theta)

        iv_args = [1]
        for i in range(len(iv_weights)):
            iv_args.append(_B_Op_(weight=-iv_weights[i], order=i+1))

        s_iv_args = [1]
        for i in range(len(seasonal_iv_weights)):
            s_iv_args.append(_B_Op_(weight=-seasonal_iv_weights[i], order=self.seasonal_periods*(i+1)))
        
        iv_op = _Dist_Brackets_(tuple(iv_args))
        siv_op = _Dist_Brackets_(tuple(s_iv_args))

        operator = iv_op * siv_op * diff_operator * seasonal_diff_operator
        operator = _Dist_Brackets_(tuple(operator.args[1:]))

        lookback = max((arg.order for arg in operator.args))
        self.w_mat = torch.zeros(lookback)
        for arg in operator.args:
            self.w_mat[-arg.order] -= arg.weight
        

    def check_kwargs(self, kwargs: dict):
        """This function checks if the keyword arguments are valid."""
        for k, v in kwargs.items():
            if k not in self.allowed_kwargs:
                raise ValueError(
                    "{key} is not a valid keyword for this model".format(key=k)
                )
        
        valid_kwargs = {}

        for k,v in kwargs.items():
            try:
                #If an exception occurs, it means kwarg is not valid
                #Hence do not set the passed value
                kwarg_data_loader = DataLoader(v)
            except:  # noqa: E722
                continue
                
            valid_kwargs[k] = kwarg_data_loader.to_tensor()

        return valid_kwargs

    def __init_model_info(self):
        """Initialize model information"""
        if self.seasonal_periods is not None:
            prefix = "S"
            seasonal_name = f"_{self.seasonal_order}{self.seasonal_periods}"
        else:
            prefix = ""
            seasonal_name = ""

        ar = "AR" if self.p + self.P!= 0 else ""
        i = "I" if self.d + self.D != 0 else ""
        ma = "MA" if self.q + self.Q != 0 else ""        
        
        model_class = prefix + ar + i + ma + f"_{self.order}" + seasonal_name
        
        self.info["model"] = model_class
        for x in [ar, i, ma]:
            if x != "":
                if x == "AR":
                    self.info["AR_Order"] = self.p

                elif x == "MA":
                    self.info["MA_Order"] = self.q

                elif x == "I":
                    self.info["Difference_Order"] = self.d    

        if prefix == "S":
            self.info["Seasonal"] = True

            self.info["Seasonal_Order"] = self.seasonal_order
            self.info["Seasonal Period"] = self.seasonal_periods

        else:
            self.info["Seasonal"] = False     

        self.info["num_params"] = sum([self.p, self.q, self.P, self.Q])

        return model_class

    def __check_arima(self):
        """Check valid ARIMA model"""

        #Orders must be >= 0 and integer
        for order in ["p", "d", "q", "P", "D", "Q"]:
            attr_val = getattr(self, order)

            if isinstance(attr_val, int) or attr_val < 0:
                try:
                    val = max(0, int(attr_val))
                except:  # noqa: E722
                    val = 0
            
                setattr(self, order, val)
        
        #Order of (0,d,0), (0,D,0) is not accepted
        if self.P + self.Q == 0:
            #setattr(self, "seasonal_periods", None)

            if self.p + self.q == 0:
                raise ValueError(
                    f"Order (0,{self.d},0) and Seasonal Order (0, {self.D}, 0) is not valid\n\
                      Include at least one seasonal or normal (or both) AR/MA component"
                )
            
            #setattr(self, "D", 0)

        #Check seasonal_periods argument is valid
        if self.seasonal_periods is not None:
            #check the seasonal_periods argument is reasonable
            if  not self.data_loader.is_valid_numeric(self.seasonal_periods):
                raise ValueError(
                    f"seasonal_periods={self.seasonal_periods} is not a valid value",
                )
            elif self.seasonal_periods < 2:
                raise ValueError(
                    "seasonal_periods must be >= 2"
                )
            else:
                #ensure seasonal periods is built_in integer
                setattr(self, "seasonal_periods", int(self.seasonal_periods))

            assert (len(self.data) > self.seasonal_periods*2),"Length\
                of data must be > 2*seasonal_periods"
        
        #If not, seasonal order is (0,0,0)
        else:
            setattr(self, "P", 0)
            setattr(self, "D", 0)
            setattr(self, "Q", 0)

        #Define the lookback for fitting to the data
        #This lookback defines how many observations we need
        #to start modeling data given the equations set by the model orders
        m = 0 if self.seasonal_periods is None else self.seasonal_periods
        lookback = self.p + self.d + m*(self.P + self.D)

        if len(self.data) < lookback:
            raise ValueError(
                f"Model with order {self.order, self.seasonal_order}\
                    needs at least {lookback} observations\n\
                    Current amount of observations: {len(self.data)}"
            ) 
        
        #TODO: Implement ADF test and recommended differencing if not stationary
    
    def calculate_confidence_interval(self, forecasts, confidence):
        """
        The Confidence Bounds for forecasts of an ARIMA model

        Bounds are given by; 
        (Chapter 5.2.3 of the book Box, G. et al. as reference)

        b_{t+n} = y_{t+n} +- z(confidence)*sqrt(1+ sum(psi[:n]**2))*var(errors)

        b_{t+n} : Probabilistic bounds for forecast at time t+n
        y_{t+n}: Forecast at time t+n
        
        z(confidence): Associated z score set by the confidence level
        var(errors): Variance of the errors generated during model fitting

        psi values are calculated by Chapter 5.2 of the book Box, G. et al. 
        as reference

        psi_j = phi_j*psi_{j-1} + phi_p*psi_{j-p} - theta_j
        psi_0 = 1

        """

        psi_vals = torch.ones(len(forecasts))
        
        for x in range(1, len(forecasts)):
            phivals = self.phi.clone()
            if self.d == 1 and len(phivals) > 0:
                ##Essentially writing; 
                #   (1 - phi_{1}*B - phi_{2}*B^2 - ...)(1 - B)Y_t as,
                #   (1 - (1 + phi_{1})B - (phi_{2} - phi_{1})B^2 - ... - (phi_{q} - phi_{q-1})B^q - (-phi_{q})B^q+1)Y_t
                nw = phivals.clone()
                nw[0] = 1 + phivals[0]
                last = -phivals[-1:]
                for j in range(1, len(phivals)):
                    nw[j] = phivals[j] - phivals[j-1]
                phivals = torch.concat((nw, last))
            
            if len(phivals) > len(psi_vals[:x]):
                phivals = phivals[:len(psi_vals[:x])]
            elif len(phivals) < len(psi_vals[:x]):
                phivals = torch.cat((phivals, torch.zeros(len(psi_vals[:x])-len(phivals))))

            psi_vals[x] = torch.sum(psi_vals[:x]*phivals.__reversed__())

            if x <= len(self.theta):
                psi_vals[x] = psi_vals[x] - self.theta[x-1]
        
        fc_variance = torch.ones(len(forecasts))
        for x in range(1, len(forecasts)):
            fc_variance[x] = torch.sqrt(1+torch.sum(torch.square(psi_vals[1:x+1])))

        z_conf = round(norm.ppf(1 - ((1 - confidence) / 2)), 2)

        upper_bounds = forecasts + z_conf*fc_variance*np.nanstd(self.fitted - self.data)
        lower_bounds = forecasts - z_conf*fc_variance*np.nanstd(self.fitted - self.data)

        return upper_bounds, lower_bounds