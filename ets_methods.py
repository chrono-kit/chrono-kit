import numpy as np

def AAdM(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta, gamma, phi = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            trend = init_trend
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = (lvl + phi*trend)*seasonal
            error = row.numpy() - y_hat
            
            seasonal_t = seasonal + (gamma*error)/(lvl + phi*trend)
            lvl = lvl + phi*trend + alpha*error/seasonal
            trend = phi*trend + beta*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + phi*trend)*seasonal
            error = row.numpy() - y_hat

            seasonal_t = seasonal + (gamma*error)/(lvl + phi*trend)
            lvl = lvl + phi*trend + alpha*error/seasonal
            trend = phi*trend + beta*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def MAdM(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta, gamma, phi = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            trend = init_trend
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = (lvl + phi*trend)*seasonal
            error = (row.numpy() - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + phi*bprev)*(1+alpha*error)
            trend = phi*bprev + beta*(lprev + phi*bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + phi*trend)*seasonal
            error = (row.numpy() - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + phi*bprev)*(1+alpha*error)
            trend = phi*bprev + beta*(lprev + phi*bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors