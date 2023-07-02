import pandas as pd
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from dataloader import DataLoader
warnings.filterwarnings("ignore")
mpl.rcParams["figure.figsize"] = (12,8)

def stl_decomposition(data, seasonal_period, loess_span = 0.2, robustness_iters = 1,show = True):
    
    data = DataLoader(data).to_tensor()#torch.clone(data).numpy()
    
    if data.ndim >= 1:

        data = np.squeeze(data)

    number_of_cycles = len(data) // seasonal_period
    
    #Creating subseries of the same place in the period
    subseries = np.array([data[i::seasonal_period] for i in range(seasonal_period)])


    # Initializing the weights and trend component
    trend = np.zeros_like(data)
    weights = np.ones(seasonal_period)
    
    for _ in range(robustness_iters): # Outer Loop
    #------------TODO
    # Apply local regression (Loess) to estimate the trend
        for i in range(seasonal_period): # Inner Loop
          pass
          #Detrend/Deseason
    
    if show:
        # Here plots of data, trend, seasonal and remainder will be drawn

        
    
    return trend, seasonal, remainder    