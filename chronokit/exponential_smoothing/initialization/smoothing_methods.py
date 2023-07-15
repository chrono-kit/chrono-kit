import numpy as np

"""ETS methods to use for initialization of parameters"""

def simple_exp(dep_var, init_components, params):

    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components          

    alpha = params[0]

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            errors.append(0)

        else:               

            y_hat = lvl
            error = row.numpy()[0] - y_hat
            

            lprev = lvl
            lvl = alpha*row.numpy()[0] + (1-alpha)*lprev

            errors.append(error)

    return errors

def holt_trend(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            trend = init_trend
            errors.append(0) 

        else:               

            y_hat = lvl + trend 
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            lvl = alpha*row.numpy()[0] + (1-alpha)*(lprev+bprev)
            trend = beta*(lvl-lprev) + (1-beta)*bprev

            errors.append(error)

    return errors

def holt_damped_trend(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta, phi = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            trend = init_trend
            errors.append(0) 

        else:               

            y_hat = lvl + phi*trend 
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            lvl = alpha*row.numpy()[0] + (1-alpha)*(lprev+phi*bprev)
            trend = beta*(lvl-lprev) + (1-beta)*phi*bprev

            errors.append(error)

    return errors

def hw_add(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta, gamma = params

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

            y_hat = lvl + trend + seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]-lprev-bprev)
            lvl = alpha*(row.numpy()[0]-seasonal) + (1-alpha)*(lprev+ bprev)
            trend = beta*(lvl - lprev) + (1-beta)*bprev

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + trend + seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]-lprev-bprev)
            lvl = alpha*(row.numpy()[0]-seasonal) + (1-alpha)*(lprev+ bprev)
            trend = beta*(lvl - lprev) + (1-beta)*bprev

            seasonals.append(seasonal_t)
            errors.append(error)         

    return errors

def hw_damped_add(dep_var, init_components, params):
    
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

            y_hat = lvl + phi*trend + seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]-lprev-phi*bprev)
            lvl = alpha*(row.numpy()[0]-seasonal) + (1-alpha)*(lprev + phi*bprev)
            trend = beta*(lvl - lprev) + (1-beta)*phi*bprev

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + phi*trend + seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]-lprev-phi*bprev)
            lvl = alpha*(row.numpy()[0]-seasonal) + (1-alpha)*(lprev + phi*bprev)
            trend = beta*(lvl - lprev) + (1-beta)*phi*bprev

            seasonals.append(seasonal_t)
            errors.append(error)             

    return errors

def hw_mul(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, beta, gamma = params

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

            y_hat = (lvl + trend)*seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]/(lprev+bprev))
            lvl = alpha*(row.numpy()[0]/seasonal) + (1-alpha)*(lprev + bprev)
            trend = beta*(lvl - lprev) + (1-beta)*bprev

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + trend)*seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]/(lprev+bprev))
            lvl = alpha*(row.numpy()[0]/seasonal) + (1-alpha)*(lprev + bprev)
            trend = beta*(lvl - lprev) + (1-beta)*bprev

            seasonals.append(seasonal_t)
            errors.append(error)            
        

    return errors

def hw_damped_mul(dep_var, init_components, params):
    
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
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]/(lprev+phi*bprev))
            lvl = alpha*(row.numpy()[0]/seasonal) + (1-alpha)*(lprev + phi*bprev)
            trend = beta*(lvl - lprev) + (1-beta)*phi*bprev

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + phi*trend)*seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = (1-gamma)*seasonal + gamma*(row.numpy()[0]/(lprev+phi*bprev))
            lvl = alpha*(row.numpy()[0]/seasonal) + (1-alpha)*(lprev + phi*bprev)
            trend = beta*(lvl - lprev) + (1-beta)*phi*bprev

            seasonals.append(seasonal_t)
            errors.append(error)                     

    return errors