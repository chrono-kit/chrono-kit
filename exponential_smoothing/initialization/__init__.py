from .ets_methods import *
from .initialization_methods import *
from .smoothing_methods import *

def get_init_method(method):

    init_method = {"heuristic": heuristic_initialization,
                    "mle": mle_initialization}[method]
    
    return init_method

def get_smooth_method(error, trend, seasonal, damped):

    if error:
        smooth_method = { 
                            (None, False, None, "add"): ets_methods.ANN,
                            (None, False, "add", "add"): ets_methods.ANA,
                            (None, False, "mul", "add"): ets_methods.ANM,
                            ("add", False, None, "add"): ets_methods.AAN,
                            ("add", False, "add", "add"): ets_methods.AAA,
                            ("add", False, "mul", "add"): ets_methods.AAM,
                            ("add", True, None, "add"): ets_methods.AAdN,
                            ("add", True, "add", "add"): ets_methods.AAdA,
                            ("add", True, "mul", "add"): ets_methods.AAdM,

                            (None, False, None, "mul"): ets_methods.MNN,
                            (None, False, "add", "mul"): ets_methods.MNA,
                            (None, False, "mul", "mul"): ets_methods.MNM,
                            ("add", False, None, "mul"): ets_methods.MAN,
                            ("add", False, "add", "mul"): ets_methods.MAA,
                            ("add", False, "mul", "mul"): ets_methods.MAM,
                            ("add", True, None, "mul"): ets_methods.MAdN,
                            ("add", True, "add", "mul"): ets_methods.MAdA,
                            ("add", True, "mul", "mul"): ets_methods.MAdM
                        }[trend, damped, seasonal, error]
    
    else:
        smooth_method = { 
                        (None, False, None): smoothing_methods.simple_exp,
                        ("add", False, None): smoothing_methods.holt_trend,
                        ("add", True, None): smoothing_methods.holt_damped_trend,
                        ("add", False, "add"): smoothing_methods.hw_add,
                        ("add", False, "mul"): smoothing_methods.hw_mul,
                        ("add", True, "add"): smoothing_methods.hw_damped_add,
                        ("add", True, "mul"): smoothing_methods.hw_damped_mul
                        }[trend, damped, seasonal]
        
    return smooth_method