import numpy as np

"""ETS methods to use for initialization of parameters"""

def ANN(dep_var, init_components, params):

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

            

            lvl = lvl + alpha*error

            errors.append(error)

    return errors

def ANA(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components      
    alpha, gamma = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = lvl +seasonal
            error = row.numpy()[0] - y_hat

            lprev = lvl

            seasonal_t = seasonal+(gamma*error)
            lvl = lprev + alpha*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + seasonal
            error = row.numpy()[0] - y_hat

            lprev = lvl

            seasonal_t = seasonal+(gamma*error)
            lvl = lprev + alpha*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def ANM(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components          
    alpha, gamma = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = lvl*seasonal
            error = row.numpy()[0] - y_hat

            lprev = lvl

            seasonal_t = seasonal+ gamma*error/lprev
            lvl = lprev + alpha*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl*seasonal
            error = row.numpy()[0] - y_hat

            lprev = lvl

            seasonal_t = seasonal+ gamma*error/lprev
            lvl = lprev + alpha*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)            

    return errors

def MNM(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, gamma = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = lvl*seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev = lvl

            seasonal_t = seasonal*(1 + gamma*error)
            lvl = lprev*(1 + alpha*error)

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl*seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev = lvl

            seasonal_t = seasonal*(1 + gamma*error)
            lvl = lprev*(1 + alpha*error)

            seasonals.append(seasonal_t)
            errors.append(error)            

    return errors

def MNA(dep_var, init_components, params):
    
    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components           
    alpha, gamma = params

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            seasonal = init_seasonals[0]
            seasonals = [seasonal]
            errors.append(0)

        elif index < seasonal_periods:            

            seasonal = init_seasonals[index]     

            y_hat = lvl + seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev = lvl

            seasonal_t = seasonal + gamma*error*(lprev+seasonal)
            lvl = lprev + alpha*error*(lprev+ seasonal)

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev = lvl

            seasonal_t = seasonal + gamma*error*(lprev+seasonal)
            lvl = lprev + alpha*error*(lprev+ seasonal)

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors


def MNN(dep_var, init_components, params):

    init_lvl, init_trend, init_seasonals, seasonal_periods = init_components            
    alpha = params[0]

    errors = []

    for index, row in enumerate(dep_var):

        if index == 0:

            lvl = init_lvl
            errors.append(0)

        else:               

            y_hat = lvl
            error = (row.numpy()[0] - y_hat)/y_hat

            lvl = lvl*(1+alpha*error)

            errors.append(error)

    return errors

def AAdA(dep_var, init_components, params):
    
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

            seasonal_t = seasonal + gamma*error
            lvl = lprev + phi*bprev + alpha*error
            trend = phi*bprev + beta*error


            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + phi*trend + seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal + gamma*error
            lvl = lprev + phi*bprev + alpha*error
            trend = phi*bprev + beta*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def MAdA(dep_var, init_components, params):
    
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
            error = (row.numpy()[0] - y_hat)/y_hat
            
            lprev, bprev = lvl, trend

            seasonal_t = seasonal + gamma*error
            lvl = lprev + phi*bprev + alpha*error*(lprev + phi*bprev + seasonal)
            trend = phi*bprev + beta*error*(lprev + phi*bprev + seasonal)

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + phi*trend + seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal + gamma*error
            lvl = lprev + phi*bprev + alpha*error*(lprev + phi*bprev + seasonal)
            trend = phi*bprev + beta*error*(lprev + phi*bprev + seasonal)

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

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
            error = row.numpy()[0] - y_hat
            
            seasonal_t = seasonal + (gamma*error)/(lvl + phi*trend)
            lvl = lvl + phi*trend + alpha*error/seasonal
            trend = phi*trend + beta*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + phi*trend)*seasonal
            error = row.numpy()[0] - y_hat

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
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + phi*bprev)*(1+alpha*error)
            trend = phi*bprev + beta*(lprev + phi*bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + phi*trend)*seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + phi*bprev)*(1+alpha*error)
            trend = phi*bprev + beta*(lprev + phi*bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def AAA(dep_var, init_components, params):
    
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

            y_hat = lvl + trend +seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal+ gamma*error
            lvl = lprev + bprev+ alpha*error
            trend = bprev + beta*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + trend +seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal+ gamma*error
            lvl = lprev + bprev+ alpha*error
            trend = bprev + beta*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def MAA(dep_var, init_components, params):
    
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
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal+gamma*error*(lprev+bprev+seasonal)
            lvl = lprev+bprev+alpha*(lprev+bprev+seasonal)*error
            trend = bprev+ beta*(lprev+bprev+seasonal)*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = lvl + trend + seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal+gamma*error*(lprev+bprev+seasonal)
            lvl = lprev+bprev+alpha*(lprev+bprev+seasonal)*error
            trend = bprev+ beta*(lprev+bprev+seasonal)*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def AAN(dep_var, init_components, params):
    
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

            lvl = lprev + bprev + alpha*error
            trend = bprev+ beta*error

            errors.append(error)

    return errors

def AAdN(dep_var, init_components, params):
    
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

            lvl = lprev + phi*bprev + alpha*error
            trend = phi*bprev+ beta*error

            errors.append(error)

    return errors


def MAN(dep_var, init_components, params):
    
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
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            lvl = (lprev+bprev)*(1+alpha*error)
            trend = bprev+ beta*(lprev+bprev)*error

            errors.append(error)

    return errors

def MAdN(dep_var, init_components, params):
    
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
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            lvl = (lprev+phi*bprev)*(1+alpha*error)
            trend = phi*bprev+ beta*(lprev+phi*bprev)*error

            errors.append(error)

    return errors

def AAM(dep_var, init_components, params):
    
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

            y_hat = (lvl + trend)* seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal + gamma*error/(lprev+bprev)
            lvl = lprev+bprev + alpha*error/seasonal
            trend = bprev+ beta*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               
            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + trend)* seasonal
            error = row.numpy()[0] - y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal + gamma*error/(lprev+bprev)
            lvl = lprev+bprev + alpha*error/seasonal
            trend = bprev+ beta*error/seasonal

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors

def MAM(dep_var, init_components, params):
    
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
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + bprev)*(1+alpha*error)
            trend = bprev + beta*(lprev + bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)            

        else:               

            seasonal = seasonals[index-seasonal_periods]  

            y_hat = (lvl + trend)*seasonal
            error = (row.numpy()[0] - y_hat)/y_hat

            lprev, bprev = lvl, trend

            seasonal_t = seasonal*(gamma*error + 1)
            lvl = (lprev + bprev)*(1+alpha*error)
            trend = bprev + beta*(lprev + bprev)*error

            seasonals.append(seasonal_t)
            errors.append(error)

    return errors