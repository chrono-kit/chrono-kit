import unittest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL as sm_stl
from chronokit.decomposition import STL
from chronokit.utils import metrics
from chronokit.preprocessing.data_transforms import MinMaxScaling

# (1,2) for testing multiplicative decomposition
scaler = MinMaxScaling(feature_range=(1,2))

passengers = pd.read_csv("../datasets/AirPassengers.csv", index_col=0)
sunspots = pd.read_csv("../datasets/Sunspots.csv", index_col=0)
temp = pd.read_csv("../datasets/monthly_temp.csv", index_col=0)

passengers = scaler.transform(passengers.values).squeeze(-1)
sunspots = scaler.transform(sunspots.values).squeeze(-1)

temp1 = scaler.transform(temp[temp.columns[0]].values)
temp2 = scaler.transform(temp[temp.columns[1]].values)

delta = 0.01

class TestSTL(unittest.TestCase):

    def test_passengers(self):

        trend, seasonal, remainder = STL(dep_var=passengers, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(passengers, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_passengers_mul(self):

        trend, seasonal, remainder = STL(dep_var=passengers, seasonal_period=12, method="mul",
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(np.log(passengers), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal))
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1(self):

        trend, seasonal, remainder = STL(dep_var=temp1, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(temp1, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1_mul(self):

        trend, seasonal, remainder = STL(dep_var=temp1, seasonal_period=12, method="mul",
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(np.log(temp1), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal))
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2(self):

        trend, seasonal, remainder = STL(dep_var=temp2, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(temp2, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2_mul(self):

        trend, seasonal, remainder = STL(dep_var=temp2, seasonal_period=12, method="mul",
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(np.log(temp2), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal))
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_sunspots(self):

        trend, seasonal, remainder = STL(dep_var=sunspots, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(sunspots, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, sm_results.trend)
        s_diff = metrics.rmse(seasonal, sm_results.seasonal)
        r_diff = metrics.rmse(remainder, sm_results.resid)
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_sunspots_mul(self):

        trend, seasonal, remainder = STL(dep_var=sunspots, seasonal_period=12, method="mul",
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(np.log(sunspots), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = metrics.rmse(trend, np.exp(sm_results.trend))
        s_diff = metrics.rmse(seasonal, np.exp(sm_results.seasonal))
        r_diff = metrics.rmse(remainder, np.exp(sm_results.resid))
        
        print(t_diff, s_diff, r_diff)
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

if __name__ == "__main__":
    unittest.main()

