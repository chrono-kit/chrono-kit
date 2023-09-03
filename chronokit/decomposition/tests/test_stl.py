import unittest
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL as sm_stl
from chronokit.decomposition import STL

passengers = pd.read_csv("/home/hasan/Desktop/Codes/yzt/datasets/AirPassengers.csv", index_col=0)
sunspots = pd.read_csv("/home/hasan/Desktop/Codes/yzt/datasets/Sunspots.csv", index_col=0)
temp = pd.read_csv("/home/hasan/Desktop/Codes/yzt/datasets/monthly_temp.csv", index_col=0)

temp1 = temp[temp.columns[0]]
temp2 = temp[temp.columns[1]]

delta = 0.5

class TestSTL(unittest.TestCase):

    def test_passengers(self):

        trend, seasonal, remainder = STL(dep_var=passengers.values, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(passengers.values.squeeze(-1), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = np.mean(np.sqrt(np.square(trend-sm_results.trend)))/np.std(trend)
        s_diff = np.mean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.std(seasonal)
        r_diff = np.mean(np.sqrt(np.square(remainder-sm_results.resid)))/np.std(remainder)
        
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp1(self):

        trend, seasonal, remainder = STL(dep_var=temp1.values, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(temp1.values, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = np.mean(np.sqrt(np.square(trend-sm_results.trend)))/np.std(trend)
        s_diff = np.mean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.std(seasonal)
        r_diff = np.mean(np.sqrt(np.square(remainder-sm_results.resid)))/np.std(remainder)
        
        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")
    
    def test_temp2(self):

        trend, seasonal, remainder = STL(dep_var=temp2.values, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(temp2.values, period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = np.mean(np.sqrt(np.square(trend-sm_results.trend)))/np.std(trend)
        s_diff = np.mean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.std(seasonal)
        r_diff = np.mean(np.sqrt(np.square(remainder-sm_results.resid)))/np.std(remainder)

        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

    def test_sunspots(self):

        trend, seasonal, remainder = STL(dep_var=sunspots.values, seasonal_period=12,
                                         degree=1, robust=True, outer_iterations=10, inner_iterations=2)
        
        sm_results = sm_stl(sunspots.values.squeeze(-1), period=12, robust=True).fit(outer_iter=10, inner_iter=2)

        t_diff = np.mean(np.sqrt(np.square(trend-sm_results.trend)))/np.std(trend)
        s_diff = np.mean(np.sqrt(np.square(seasonal-sm_results.seasonal)))/np.std(seasonal)
        r_diff = np.mean(np.sqrt(np.square(remainder-sm_results.resid)))/np.std(remainder)

        
        self.assertLessEqual(t_diff, delta, "Trend difference is too big")
        self.assertLessEqual(s_diff, delta, "Seasonal difference is too big")
        self.assertLessEqual(r_diff, delta, "Remainder difference is too big")

if __name__ == "__main__":
    unittest.main()

