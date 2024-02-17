import numpy as np
import torch
from scipy.stats import norm
from chronokit.preprocessing._dataloader import DataLoader
from chronokit.base._models import TraditionalTimeSeriesModel
from chronokit.exponential_smoothing.__initialization import SmoothingInitializer

class InnovationsStateSpace(TraditionalTimeSeriesModel):

    def __init__(self, data, initialization_method="heuristic", **kwargs):
        """
        Base model class for state space models

        InnovationsStateSpace name was inspired from Hyndman, R. et al. [1]

        General state space model equation can be defined as;

            y_t = w(x_{t-1}) + r(x_{t-1})*e_t
            x_t = f(x_{t-1}) + g(x_{t-1})*e_t

        where x_t is the state vector at time t.

        All methods have been written with the below book as reference: 
            [1] 'Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). 
            Forecasting with exponential smoothing: The state space approach'
        
        This book may also be referenced as 'Hyndman, R. et al." throughout this repository
        """

        super().__init__(data, **kwargs)
        
        #Check valid model
        self.__check_innovations_state_space()
        self.__init_model_info()

        if not len(data) >= 10 and initialization_method != "known": 
            raise Warning(
                "Initialization will fail for num observations < 10,\
                please pass known parameters and use initialization=known"
            )

        if self.seasonal is not None:
            # Data must have at least 2 full seasonal cycles
            assert (len(data) > 2 * self.seasonal_periods), "Length of data\
                must be > 2*seasonal_periods for a seasonal model"

        if initialization_method != "known":
            initializer = SmoothingInitializer(self, method=initialization_method)

            init_params, init_components = initializer.initialize_parameters()

            self.alpha = init_params["alpha"]
            self.beta = init_params["beta"]
            self.gamma = init_params["gamma"]
            self.phi = init_params["phi"]

            self.level = init_components["level"]
            self.trend_factor = init_components["trend_factor"]
            self.seasonal_factors = init_components["seasonal_factors"]

            self.info["optimization_success"] = initializer.success if initializer is not None else "nan"

        else:
            if not set(self.allowed_kwargs).issubset(kwargs.keys()):
                raise Exception(
                    f"All values component should be passes when \
                    initialization_method='known'\n {self.allowed_kwargs}"
                )
                

        self.set_kwargs(kwargs)
            
        self.params = {"alpha": self.alpha,
                        "beta": self.beta,
                        "gamma": self.gamma,
                        "phi": self.phi}

        self.init_components = {"level": self.level,
                                "trend_factor": self.trend_factor,
                                "seasonal_factors": self.seasonal_factors}
        
        self.seasonal_factors = DataLoader(self.seasonal_factors).to_tensor()
        self.level = DataLoader(self.level).to_tensor()
        self.trend_factor = DataLoader(self.trend_factor).to_tensor()

        self.fitted = torch.zeros(size=self.data.shape)
        self.errors = torch.zeros(size=self.data.shape)
    
    def calculate_confidence_interval(self, forecasts, confidence):
        """
        The calculations are made according to Table 6.1 in Hyndman, R. et al.
        For model classes 1-3, we use given expressions in chapter 6.
        For model classes 4-5, we use simulated intervals.
        """

        if "error" in self.info:
            # TODO: change this statement when class 3 algebraic approximation is implemented
            if self.info["trend"] != "M" and self.info["seasonal"] != "M":
                fc_variance = self._confidence_interval_known_method_ets(forecasts)
                z_conf = round(norm.ppf(1 - ((1 - confidence) / 2)), 2)

                upper_bounds = forecasts + z_conf*torch.sqrt(fc_variance)
                lower_bounds = forecasts - z_conf*torch.sqrt(fc_variance)

            else:
                bounds = super()._confidence_interval_simulation_method(h=len(forecasts), confidence=confidence)

                upper_bounds = forecasts + bounds
                lower_bounds = forecasts - bounds
        else:
            bounds = super()._confidence_interval_simulation_method(h=len(forecasts), confidence=confidence)

            upper_bounds = forecasts + bounds
            lower_bounds = forecasts - bounds
        
        return upper_bounds, lower_bounds


    def _confidence_interval_known_method_ets(self, forecasts):
        
        """
        See Hyndman, R. et al. Chapter 6
        Approximate intervals for model class 3 
        has not been implemented yet, so we use simulation.
        Algebraic methods for classes 1,2 are provided here
        """

        #We need point forecasts for some methods, so instead of taking h as an argument;
        #We infer it by length of the point forecasts
        h = len(forecasts)

        #See BOOK 6.5
        #We take the errors directly from those recorded during training
        #As we need to use the multiplicative errors for ETS(M,X,X)
        fit_errors = self.errors
        error_var = torch.nanmean(torch.square(fit_errors))

        #Class 1-3 share c_j values

        c_values = torch.zeros(h)

        for step in range(1, h+1):

            damp_factor = torch.sum(DataLoader([self.phi**x for x in range(1, step+1)]).to_tensor())

            cj = self.alpha + self.alpha*self.beta*damp_factor

            if self.info["seasonal"] == "A":
                if step % self.seasonal_periods == 0:
                    cj += self.gamma
                
            c_values[step-1] = cj

        #Class 1-2
        if self.info["seasonal"] != "M":
            
            #Class 1
            if self.info["error"] == "A":

                var_multipliers = torch.ones(h)

                if h == 1:
                    return error_var*var_multipliers

                for x in range(2, h+1):
                    var_multipliers[x-1] = 1 + torch.sum(torch.square(c_values[:x-1]))
                
                return error_var*var_multipliers

            #Class 2
            elif self.info["error"] == "M":
                
                var_multipliers = torch.square(forecasts.clone())
                if h == 1:
                    return error_var*var_multipliers

                for x in range(2, h+1):
                    #this is tricky, leave notes from the equation provided by the book
                    previous_multipliers = reversed(var_multipliers[:x-1].clone())
                    #print(previous_multipliers)
                    var_multipliers[x-1] = var_multipliers[x-1] + error_var*torch.sum(torch.square(c_values[:x-1])*previous_multipliers)

                return (1+error_var)*var_multipliers - torch.square(forecasts.clone())
        
        #TODO: implement class 3,
        #Until implementation, the above if statement should yield 
        #True for all times when this function is called
        else:
            return None

    def __init_model_info(self):

        if hasattr(self, "error_type"):
            prefix = "ETS"

            error = self.error_type[0].upper()
            trend = self.trend[0].upper() if self.trend is not None else "N"
            if self.damped and trend != "N":
                trend = trend + "d"
            seasonal = self.seasonal[0].upper() if self.seasonal is not None else "N"

            model_class = prefix + f"({error}, {trend}, {seasonal})"

            self.info["model"] = model_class 
            self.info["error"] = error
            self.info["trend"] = trend
            self.info["seasonal"] = seasonal
            if seasonal != "N":
                self.info["seasonal_periods"] = str(self.seasonal_periods)
        
        elif hasattr(self, "trend"):
            prefix = "HoltWinters"

            trend = self.trend[0].upper() if self.trend is not None else "N"
            if self.damped and trend != "N":
                trend = trend + "d"
            seasonal = self.seasonal[0].upper() if self.seasonal is not None else "N"

            model_class = prefix + f"({trend}, {seasonal})"

            self.info["model"] = model_class 
            self.info["trend"] = trend
            self.info["seasonal"] = seasonal
            if seasonal != "N":
                self.info["seasonal_periods"] = str(self.seasonal_periods)

        else:
            #This is a placeholder for possible future implementations 
            #of generalized state space models
            model_class = "NONE"
        
        self.info["num_params"] = 1 + sum([bool(self.trend), bool(self.seasonal), self.damped])
    
    def __check_innovations_state_space(self):
        
        if self.seasonal not in ["add", "mul"]:
            setattr(self, "seasonal", None)
        
        else:
            #check the seasonal_periods argument is reasonable
            if not self.data_loader.is_valid_numeric(self.seasonal_periods):
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

        if self.trend not in ["add", "mul"]:
            setattr(self, "trend", None)
        
        if self.damped not in [True, False]:
            setattr(self, "damped", False)

        #Models with mul components need strictly positive data
        
        if hasattr(self, "error_type"):
            if self.error_type not in ["add", "mul"]:
                setattr(self, "error_type", "add")
            
            if self.error_type == "mul":
                assert (
                    self.data_loader.to_numpy().min() > 0
                ), "Methods with multiplicative components requires \
                    data to be strictly positive"

        if self.trend == "mul" or self.seasonal == "mul":
            assert (
                self.data_loader.to_numpy().min() > 0
            ), "Methods with multiplicative components requires \
                data to be strictly positive"

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
            
            valid = True

            #seasonal_factors require different handling
            #as it is expected to be an array_like instead of numeric
            if "seasonal" in k:
                seasonal_factors = kwarg_data_loader.to_tensor()
                
                #Try iterating over provided seasonal factors
                #If cannot iterate or at least one value is not valid
                #Do not set the seasonal_factors by the kwarg
                try:
                    for x in iter(seasonal_factors):
                        if not kwarg_data_loader.is_valid_numeric(x):
                            valid = False
                except:  # noqa: E722
                    valid = False
            
            else:
                if not kwarg_data_loader.is_valid_numeric(v):
                    valid = False
                
                #These parameters should also need to be in range (0,1)
                if k in ["alpha", "beta", "gamma", "phi"]:
                    if not (v >= 0 and v <= 1):
                        valid = False
                
            if valid:
                #remove the initial when setting passed values for kwargs
                if "initial" in k:
                    #remove the "initial_" from the component name
                    k = "_".join(k.split("_")[1:])
                valid_kwargs[k] = v

        return valid_kwargs
        
    def update_state(self):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")
    
    def fit(self):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")

    def predict(self, h: int, confidence: float = None):
        # This function will be overriden by the child class.
        raise NotImplementedError("This function is not implemented yet.")