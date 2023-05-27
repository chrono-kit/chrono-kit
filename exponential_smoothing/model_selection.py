import torch
import numpy as np 
from .initialization import ets_methods, smoothing_methods
from exponential_smoothing import ETS, ExponentialSmoothing
from dataloader import DataLoader
from metrics import Metric
import time
import pandas as pd


#keys are "trend, damped, seasonal, error"
all_models =  [
                (None, False, None, None),
                ("add", False, None, None),
                ("add", True, None, None),
                ("add", False, "add", None),
                ("add", False, "mul", None),
                ("add", True, "add", None),
                ("add", True, "mul", None),

                (None, False, None, "add"),
                (None, False, "add", "add"),
                (None, False, "mul", "add"),
                ("add", False, None, "add"),
                ("add", False, "add", "add"),
                ("add", False, "mul", "add"),
                ("add", True, None, "add"),
                ("add", True, "add", "add"),
                ("add", True, "mul", "add"),

                (None, False, None, "mul"),
                (None, False, "add", "mul"),
                (None, False, "mul", "mul"),
                ("add", False, None, "mul"),
                ("add", False, "add", "mul"),
                ("add", False, "mul", "mul"),
                ("add", True, None, "mul"),
                ("add", True, "add", "mul"),
                ("add", True, "mul", "mul")
                                            ]

def model_selection(data, seasonal_periods=12, criterion="rmse"):

    data = DataLoader(data).to_tensor()

    trend_args = [None, "add"]
    damped_args = [False, True]
    seasonal_args = [None, "add", "mul"]
    error_args = [None, "add", "mul"]

    results_dict = {}

    if criterion == "rmse":

        for e_arg in error_args:

            if e_arg is None:

                for t_arg in trend_args:
                    for d_arg in damped_args:
                        for s_arg in seasonal_args:

                            if (t_arg, d_arg, s_arg, e_arg) not in all_models:
                                continue
                            else:
                                model = ExponentialSmoothing(dep_var=data, trend=t_arg, damped=d_arg,
                                                             seasonal=s_arg, seasonal_periods=12)
                            
                                start_time = time.perf_counter()

                                model.fit()

                                end_time = time.perf_counter()
                                fit_time = end_time - start_time

                                rmse = Metric.rmse(model.fitted, data)
                                
                                results_dict[(t_arg, d_arg, s_arg, e_arg)] = {"RMSE": f"{rmse.numpy():.4f}", "Fit Time(Seconds)": f"{fit_time:.4f}"}
            
            else:

                for t_arg in trend_args:
                    for d_arg in damped_args:
                        for s_arg in seasonal_args:

                            if (t_arg, d_arg, s_arg, e_arg) not in all_models:
                                continue
                            else:
                                model = ETS(dep_var=data,trend=t_arg, damped=d_arg, error_type=e_arg,
                                                seasonal=s_arg, seasonal_periods=12)
                            
                                start_time = time.perf_counter()

                                model.fit()

                                end_time = time.perf_counter()
                                fit_time = end_time - start_time

                                rmse = Metric.rmse(model.fitted, data)
                                
                                results_dict[(t_arg, d_arg, s_arg, e_arg)] = {"RMSE": f"{rmse.numpy():.4f}", "Fit Time(Seconds)": f"{fit_time:.4f}"}

        rmse_list = []
        time_list = []
        index = []

        for key in results_dict:

            index.append(f"{key}")

            rmse_list.append(results_dict[key]["RMSE"])
            time_list.append(results_dict[key]["Fit Time(Seconds)"])

        print_df = pd.DataFrame(rmse_list, columns=["RMSE"], index=index)
        print_df["Fit Time(Seconds)"] = time_list
        print_df.index.name = "Args(Trend, Damped, Seasonal, Error)"
        print(print_df)

        best_rmse = 1e6
        best_args = {"trend": None, "damped": None, "seasonal": None, "error": None}

        for args in results_dict:

            rmse = float(results_dict[args]["RMSE"])

            if rmse < best_rmse:

                best_rmse = rmse

                best_args["trend"] = args[0]
                best_args["damped"] = args[1]
                best_args["seasonal"] = args[2]
                best_args["error"] = args[3]

        return best_args
            




